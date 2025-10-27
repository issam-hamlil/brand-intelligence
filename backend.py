"""
Production Backend - 100% Free Stack - DOCKER OPTIMIZED
Works in Docker containers with proper service discovery
"""

import os
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import time

# Docker-aware configuration
IS_DOCKER = os.path.exists('/.dockerenv')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://ollama:11434' if IS_DOCKER else 'http://localhost:11434')

print(f"🐳 Running in Docker: {IS_DOCKER}")
print(f"🔗 Ollama host: {OLLAMA_HOST}")

# ============================================
# FREE LLM - Docker-Compatible Ollama Client
# ============================================
class OllamaClient:
    """Docker-aware Ollama client with retries and health checks"""
    
    def __init__(self, base_url: str = OLLAMA_HOST, model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.max_retries = 5
        self.retry_delay = 2
        
        # Wait for Ollama to be ready
        self._wait_for_ollama()
        
        # Check if model exists
        self._ensure_model()
    
    def _wait_for_ollama(self):
        """Wait for Ollama service to be ready"""
        import requests
        
        print(f"🔍 Waiting for Ollama at {self.base_url}...")
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Ollama is ready!")
                    return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⏳ Attempt {attempt + 1}/{self.max_retries}: Waiting for Ollama... ({e})")
                    time.sleep(self.retry_delay)
                else:
                    print(f"❌ Ollama not available after {self.max_retries} attempts")
                    raise Exception(f"Ollama service not available at {self.base_url}")
    
    def _ensure_model(self):
        """Check if model exists, provide clear instructions if not"""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if any(self.model in name for name in model_names):
                    print(f"✅ Model '{self.model}' found!")
                    return
                else:
                    print(f"⚠️  Model '{self.model}' not found!")
                    print(f"📥 Available models: {model_names}")
                    print(f"\n🔧 To install, run:")
                    print(f"   docker exec ollama ollama pull {self.model}")
                    
                    # Use fallback persona generation
                    self.model = None
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            self.model = None
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        """Generate text with Docker-compatible settings"""
        import requests
        
        if not self.model:
            print("⚠️  No model available, using fallback")
            return ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=120  # Longer timeout for Docker
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""

# ============================================
# FREE SENTIMENT - HuggingFace Transformers
# ============================================
class FreeSentimentAnalyzer:
    """Docker-optimized sentiment analysis with caching"""
    
    def __init__(self, use_gpu: bool = False):
        print("🔄 Loading HuggingFace models (this may take a minute first time)...")
        
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        
        device = 0 if use_gpu else -1
        
        # Cache directory for Docker volumes
        cache_dir = "/app/.cache" if IS_DOCKER else None
        
        try:
            # Free sentiment model
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
                truncation=True,
                max_length=512,
                cache_dir=cache_dir
            )
            
            # Free emotion model
            self.emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=device,
                truncation=True,
                max_length=512,
                cache_dir=cache_dir
            )
            
            # Free embeddings
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
            
            print("✅ All models loaded!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def analyze(self, text: str) -> Dict:
        """Analyze text with error handling"""
        try:
            # Truncate
            text = text[:512] if text else "neutral text"
            
            # Sentiment
            sentiment = self.sentiment_pipe(text)[0]
            sentiment_score = self._normalize_sentiment(sentiment)
            
            # Emotion
            emotion = self.emotion_pipe(text)[0]
            
            # Embedding
            embedding = self.embedder.encode(text).tolist()
            
            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment['label'],
                "emotion": emotion['label'],
                "emotion_score": emotion['score'],
                "embedding": embedding
            }
        except Exception as e:
            print(f"⚠️  Analysis error for text: {e}")
            # Return neutral fallback
            return {
                "sentiment_score": 0.5,
                "sentiment_label": "neutral",
                "emotion": "neutral",
                "emotion_score": 0.5,
                "embedding": [0.0] * 384
            }
    
    def _normalize_sentiment(self, result: Dict) -> float:
        """Convert to 0-1 scale"""
        label = result['label'].lower()
        score = result['score']
        
        if 'positive' in label:
            return 0.5 + (score * 0.5)
        elif 'negative' in label:
            return 0.5 - (score * 0.5)
        else:
            return 0.5
    
    def batch_analyze(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Process multiple texts efficiently"""
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.analyze(text))
            
            # Progress indicator
            processed = min(i + batch_size, total)
            print(f"  📊 Analyzed {processed}/{total} mentions...")
        
        return results

# ============================================
# PERSONA GENERATOR - Docker Optimized
# ============================================
class PersonaGenerator:
    """Generate personas with Docker-aware fallbacks"""
    
    def __init__(self):
        self.llm = OllamaClient(base_url=OLLAMA_HOST)
        self.embedder = None
    
    def _init_embedder(self):
        """Lazy load embedder"""
        if not self.embedder:
            from sentence_transformers import SentenceTransformer
            cache_dir = "/app/.cache" if IS_DOCKER else None
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    
    def generate_personas(
        self, 
        brand_keywords: List[str], 
        num_personas: int = 3,
        regional_data: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate personas with robust error handling"""
        
        print(f"🎭 Generating {num_personas} personas...")
        
        # Build prompt
        context = self._build_context(regional_data)
        prompt = self._build_prompt(brand_keywords, num_personas, context)
        
        # Try LLM generation
        response = self.llm.generate(
            prompt=prompt,
            system="You are a marketing expert. Output ONLY valid JSON array.",
            max_tokens=1500
        )
        
        # Parse response
        personas = self._parse_response(response, brand_keywords, num_personas)
        
        # Add embeddings
        personas = self._add_embeddings(personas)
        
        print(f"✅ Generated {len(personas)} personas")
        return personas
    
    def _build_context(self, regional_data: Optional[Dict]) -> str:
        """Build context from regional data"""
        if not regional_data:
            return ""
        
        context = "\n\nRegional Context:\n"
        for region in regional_data:
            context += f"- {region.get('region', 'unknown')}: "
            context += f"sentiment {region.get('avg_sentiment', 0.5):.2f}, "
            context += f"{region.get('mention_count', 0)} mentions\n"
        
        return context
    
    def _build_prompt(self, keywords: List[str], num: int, context: str) -> str:
        """Create structured prompt"""
        keywords_str = ', '.join(keywords)
        
        return f"""Create {num} distinct customer personas for a brand with these keywords: {keywords_str}
{context}

For each persona, provide JSON with exactly these fields:
- name: Creative name (e.g., "Eco-Emma")
- age_range: Age bracket (e.g., "25-34")
- interests: Array of 3 interests
- motivation: One sentence motivation
- tone: Recommended ad tone
- ad_sample: One compelling ad sentence

Output ONLY a JSON array. Example:
[{{"name": "Sustainable Sarah", "age_range": "28-35", "interests": ["Eco-products", "Ethical shopping", "Sustainability"], "motivation": "Support sustainable businesses", "tone": "Authentic and educational", "ad_sample": "Make every purchase count for the planet."}}]

Generate {num} personas now as JSON array:"""
    
    def _parse_response(self, response: str, keywords: List[str], num: int) -> List[Dict]:
        """Parse LLM response with fallbacks"""
        if not response:
            print("⚠️  Empty response, using fallback personas")
            return self._fallback_personas(keywords, num)
        
        try:
            # Extract JSON
            if '[' in response and ']' in response:
                start = response.index('[')
                end = response.rindex(']') + 1
                personas = json.loads(response[start:end])
                
                if isinstance(personas, list) and len(personas) > 0:
                    # Validate structure
                    valid_personas = []
                    for p in personas[:num]:
                        if all(k in p for k in ['name', 'interests', 'motivation']):
                            valid_personas.append(p)
                    
                    if valid_personas:
                        print(f"✅ Parsed {len(valid_personas)} personas from LLM")
                        return valid_personas
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
        
        print("⚠️  Using fallback personas")
        return self._fallback_personas(keywords, num)
    
    def _fallback_personas(self, keywords: List[str], num: int) -> List[Dict]:
        """High-quality fallback personas"""
        all_personas = [
            {
                "name": "Eco-Conscious Emma",
                "age_range": "28-35",
                "interests": ["Sustainability", "Organic Products", "Climate Action"],
                "motivation": "Reduce environmental impact through purchasing decisions",
                "tone": "Educational, authentic, values-driven",
                "ad_sample": "Every purchase is a vote for the planet. Choose wisely. 🌱"
            },
            {
                "name": "Budget-Smart Brian",
                "age_range": "35-50",
                "interests": ["Value Shopping", "Family Budget", "Smart Deals"],
                "motivation": "Maximize value without sacrificing quality for family",
                "tone": "Straightforward, trustworthy, practical",
                "ad_sample": "Quality products at honest prices. Your family deserves both."
            },
            {
                "name": "Tech-Forward Taylor",
                "age_range": "22-32",
                "interests": ["Innovation", "Early Adoption", "Smart Technology"],
                "motivation": "Stay ahead with cutting-edge products and experiences",
                "tone": "Exciting, modern, aspirational",
                "ad_sample": "The future is here. Be among the first to experience it. ⚡"
            },
            {
                "name": "Quality-First Quinn",
                "age_range": "40-55",
                "interests": ["Premium Products", "Durability", "Craftsmanship"],
                "motivation": "Invest in lasting quality over cheap alternatives",
                "tone": "Sophisticated, refined, timeless",
                "ad_sample": "Excellence never goes out of style. Invest in forever."
            },
            {
                "name": "Community-Minded Morgan",
                "age_range": "30-45",
                "interests": ["Local Business", "Social Impact", "Community Support"],
                "motivation": "Support businesses that give back to community",
                "tone": "Warm, inclusive, community-focused",
                "ad_sample": "Together, we build stronger communities. Join us today."
            }
        ]
        
        return all_personas[:num]
    
    def _add_embeddings(self, personas: List[Dict]) -> List[Dict]:
        """Add embeddings for persona-region matching"""
        self._init_embedder()
        
        for persona in personas:
            try:
                # Create text representation
                interests_text = ' '.join(persona.get('interests', []))
                motivation_text = persona.get('motivation', '')
                text = f"{interests_text} {motivation_text}"
                
                # Generate embedding
                embedding = self.embedder.encode(text).tolist()
                persona['embedding'] = embedding
            except Exception as e:
                print(f"⚠️  Embedding error: {e}")
                persona['embedding'] = [0.0] * 384  # Fallback
        
        return personas

# ============================================
# DATABASE - Docker Optimized SQLite
# ============================================
class BrandIntelligenceDB:
    """Docker-friendly database with proper paths"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use /app for Docker persistence
            db_path = "/app/brand_intelligence.db" if IS_DOCKER else "brand_intelligence.db"
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_schema()
        print(f"✅ Database ready: {db_path}")
    
    def _setup_schema(self):
        """Create optimized schema"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT DEFAULT 'csv',
                region TEXT NOT NULL,
                sentiment_score REAL,
                sentiment_label TEXT,
                emotion TEXT,
                emotion_score REAL,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_region ON mentions(region);
            CREATE INDEX IF NOT EXISTS idx_sentiment ON mentions(sentiment_score);
            
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age_range TEXT,
                interests TEXT,
                motivation TEXT,
                tone TEXT,
                ad_sample TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS persona_region_matches (
                persona_id INTEGER,
                region TEXT,
                match_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (persona_id) REFERENCES personas(id)
            );
        """)
        self.conn.commit()
    
    def add_mention(self, text: str, region: str, analysis: Dict):
        """Store analyzed mention"""
        try:
            self.conn.execute("""
                INSERT INTO mentions (text, region, sentiment_score, sentiment_label, emotion, emotion_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                text,
                region,
                analysis['sentiment_score'],
                analysis['sentiment_label'],
                analysis['emotion'],
                analysis['emotion_score'],
                json.dumps(analysis['embedding'])
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️  DB insert error: {e}")
    
    def add_persona(self, persona: Dict) -> int:
        """Store persona and return ID"""
        try:
            cursor = self.conn.execute("""
                INSERT INTO personas (name, age_range, interests, motivation, tone, ad_sample, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                persona['name'],
                persona.get('age_range', ''),
                json.dumps(persona.get('interests', [])),
                persona.get('motivation', ''),
                persona.get('tone', ''),
                persona.get('ad_sample', ''),
                json.dumps(persona.get('embedding', []))
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"⚠️  DB insert error: {e}")
            return -1
    
    def get_regional_summary(self) -> pd.DataFrame:
        """Get aggregated regional data"""
        query = """
            SELECT 
                region,
                COUNT(*) as mention_count,
                AVG(sentiment_score) as avg_sentiment,
                GROUP_CONCAT(DISTINCT emotion) as emotions
            FROM mentions
            GROUP BY region
            ORDER BY mention_count DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def match_personas_to_regions(self) -> List[Dict]:
        """Match personas to best regions using embeddings"""
        personas = self.conn.execute("SELECT * FROM personas").fetchall()
        mentions = self.conn.execute("SELECT region, embedding, sentiment_score FROM mentions").fetchall()
        
        matches = []
        
        for persona in personas:
            persona_id = persona[0]
            persona_name = persona[1]
            
            try:
                persona_emb = np.array(json.loads(persona[7]))
            except:
                continue
            
            region_scores = {}
            
            for mention in mentions:
                region = mention[0]
                try:
                    mention_emb = np.array(json.loads(mention[1]))
                    sentiment = mention[2]
                except:
                    continue
                
                # Cosine similarity
                norm_p = np.linalg.norm(persona_emb)
                norm_m = np.linalg.norm(mention_emb)
                
                if norm_p > 0 and norm_m > 0:
                    similarity = np.dot(persona_emb, mention_emb) / (norm_p * norm_m)
                    weighted_score = similarity * (0.5 + sentiment * 0.5)
                    
                    if region not in region_scores:
                        region_scores[region] = []
                    region_scores[region].append(weighted_score)
            
            # Average scores per region
            region_averages = [
                {"region": r, "score": float(np.mean(scores))}
                for r, scores in region_scores.items()
            ]
            region_averages.sort(key=lambda x: x['score'], reverse=True)
            
            # Store top matches
            for match in region_averages[:3]:
                self.conn.execute("""
                    INSERT INTO persona_region_matches (persona_id, region, match_score)
                    VALUES (?, ?, ?)
                """, (persona_id, match['region'], match['score']))
            
            matches.append({
                "persona": persona_name,
                "best_regions": region_averages[:3]
            })
        
        self.conn.commit()
        return matches
    
    def get_persona_details(self) -> List[Dict]:
        """Get all personas with their best regions"""
        query = """
            SELECT 
                p.*,
                GROUP_CONCAT(prm.region || ':' || prm.match_score) as region_matches
            FROM personas p
            LEFT JOIN persona_region_matches prm ON p.id = prm.persona_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """
        
        rows = self.conn.execute(query).fetchall()
        
        personas = []
        for row in rows:
            region_matches = []
            if row[8]:
                for match_str in row[8].split(','):
                    try:
                        region, score = match_str.split(':')
                        region_matches.append({
                            "region": region,
                            "score": float(score)
                        })
                    except:
                        continue
            
            personas.append({
                "id": row[0],
                "name": row[1],
                "age_range": row[2],
                "interests": json.loads(row[3]) if row[3] else [],
                "motivation": row[4],
                "tone": row[5],
                "ad_sample": row[6],
                "best_regions": region_matches
            })
        
        return personas

# ============================================
# MAIN PIPELINE - Docker Optimized
# ============================================
class BrandIntelligencePipeline:
    """Complete pipeline with Docker optimizations"""
    
    def __init__(self, db_path: str = None):
        print("\n" + "="*60)
        print("🐳 BRAND INTELLIGENCE PIPELINE - DOCKER MODE")
        print("="*60 + "\n")
        
        self.db = BrandIntelligenceDB(db_path)
        self.sentiment_analyzer = FreeSentimentAnalyzer()
        self.persona_generator = PersonaGenerator()
    
    def process_mentions(self, csv_path: str) -> Dict:
        """Process mentions from CSV"""
        print(f"\n📊 Loading mentions from {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            raise
        
        required_cols = ['text', 'region']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        print(f"🔍 Analyzing {len(df)} mentions...")
        
        for idx, row in df.iterrows():
            analysis = self.sentiment_analyzer.analyze(row['text'])
            self.db.add_mention(row['text'], row['region'], analysis)
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                print(f"  ✓ Processed {idx + 1}/{len(df)} mentions")
        
        print(f"✅ All mentions analyzed!")
        
        return {
            "total_mentions": len(df),
            "regions": df['region'].nunique()
        }
    
    def generate_personas(
        self, 
        brand_keywords: List[str], 
        num_personas: int = 3
    ) -> List[Dict]:
        """Generate personas"""
        # Get regional context
        regional_summary = self.db.get_regional_summary()
        regional_data = regional_summary.to_dict('records') if not regional_summary.empty else None
        
        # Generate personas
        personas = self.persona_generator.generate_personas(
            brand_keywords,
            num_personas,
            regional_data
        )
        
        # Store in DB
        for persona in personas:
            self.db.add_persona(persona)
        
        return personas
    
    def match_personas_to_regions(self) -> List[Dict]:
        """Create persona-region matches"""
        print("\n🎯 Matching personas to regions...")
        matches = self.db.match_personas_to_regions()
        print("✅ Matching complete!")
        return matches
    
    def get_full_report(self) -> Dict:
        """Get complete intelligence report"""
        return {
            "regional_summary": self.db.get_regional_summary().to_dict('records'),
            "personas": self.db.get_persona_details()
        }
    
    def run_full_pipeline(
        self, 
        csv_path: str, 
        brand_keywords: List[str],
        num_personas: int = 3
    ) -> Dict:
        """Run complete pipeline end-to-end"""
        
        # Step 1: Process mentions
        mention_stats = self.process_mentions(csv_path)
        
        # Step 2: Generate personas
        print(f"\n🎭 Generating {num_personas} personas...")
        personas = self.generate_personas(brand_keywords, num_personas)
        
        # Step 3: Match personas to regions
        matches = self.match_personas_to_regions()
        
        # Step 4: Get full report
        report = self.get_full_report()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print(f"📊 Processed: {mention_stats['total_mentions']} mentions")
        print(f"📍 Regions: {mention_stats['regions']}")
        print(f"🎭 Personas: {len(personas)}")
        print("="*60 + "\n")
        
        return report

# ============================================
# CLI INTERFACE
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Free Brand Intelligence Pipeline")
    parser.add_argument("--csv", required=True, help="Path to mentions CSV")
    parser.add_argument("--keywords", required=True, help="Comma-separated brand keywords")
    parser.add_argument("--personas", type=int, default=3, help="Number of personas")
    parser.add_argument("--output", default="report.json", help="Output report path")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = BrandIntelligencePipeline()
    report = pipeline.run_full_pipeline(
        csv_path=args.csv,
        brand_keywords=[k.strip() for k in args.keywords.split(',')],
        num_personas=args.personas
    )
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to: {args.output}")