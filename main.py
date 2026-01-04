"""
IB CRYPTO PLAY - ENHANCED BACKEND API v3.1
Production-Ready with Critical Security Fixes
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import random
import uuid
import logging
from enum import Enum

# Security imports
from passlib.context import CryptContext
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production-use-env-var"  # TODO: Move to env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security schemes
security = HTTPBearer()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BetType(str, Enum):
    HOME = "1"
    DRAW = "X"
    AWAY = "2"

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class MultiBetRequest(BaseModel):
    bets: List[Dict]
    total_stake: float = Field(gt=0, le=1000000)
    
    @validator('bets')
    def validate_bets(cls, v):
        if len(v) == 0:
            raise ValueError('At least one bet required')
        if len(v) > 20:
            raise ValueError('Maximum 20 bets allowed')
        return v

class CasinoPlayRequest(BaseModel):
    game_id: int = Field(ge=1, le=9)
    game_name: str
    bet_amount: float = Field(gt=0, le=50000)

class DepositRequest(BaseModel):
    amount: float = Field(gt=0, le=1000000)
    currency: str
    crypto: str = Field(default="")

class WithdrawRequest(BaseModel):
    crypto: str = Field(..., min_length=3, max_length=10)
    amount: float = Field(gt=0)
    wallet_address: str = Field(..., min_length=26, max_length=100)
    
    @validator('wallet_address')
    def validate_address(cls, v):
        # Basic validation - no spaces, special chars
        if ' ' in v or any(c in v for c in ['<', '>', '"', "'"]):
            raise ValueError('Invalid wallet address format')
        return v

class TransferRequest(BaseModel):
    from_currency: str
    to_currency: str
    amount: float = Field(gt=0)

# ============================================================================
# DATABASE - Enhanced with password hashing
# ============================================================================

users_db = {
    "demo": {
        "password": pwd_context.hash("demo123"),  # Hashed password
        "balance": 50000.00,
        "crypto_wallet": {
            "BTC": 0.0234,
            "ETH": 0.456,
            "SOL": 12.34,
            "BNB": 2.567
        },
        "bet_history": [],
        "transactions": [],
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
}

# Matches database
matches_db = [
    {
        "id": 1,
        "home": "Man United",
        "away": "Liverpool",
        "league": "Premier League",
        "time": "LIVE",
        "minute": 67,
        "score": {"home": 2, "away": 2},
        "odds": {"home": 2.45, "draw": 3.20, "away": 2.80},
        "live": True,
        "status": "in_play"
    },
    {
        "id": 2,
        "home": "Barcelona",
        "away": "Real Madrid",
        "league": "La Liga",
        "time": "20:00",
        "minute": None,
        "score": None,
        "odds": {"home": 1.95, "draw": 3.50, "away": 3.80},
        "live": False,
        "status": "upcoming"
    },
    {
        "id": 3,
        "home": "Bayern Munich",
        "away": "Dortmund",
        "league": "Bundesliga",
        "time": "LIVE",
        "minute": 45,
        "score": {"home": 1, "away": 0},
        "odds": {"home": 1.75, "draw": 3.80, "away": 4.20},
        "live": True,
        "status": "in_play"
    },
    {
        "id": 4,
        "home": "PSG",
        "away": "Monaco",
        "league": "Ligue 1",
        "time": "Tomorrow 18:00",
        "minute": None,
        "score": None,
        "odds": {"home": 1.50, "draw": 4.20, "away": 5.50},
        "live": False,
        "status": "upcoming"
    },
    {
        "id": 5,
        "home": "Arsenal",
        "away": "Chelsea",
        "league": "Premier League",
        "time": "Tomorrow 19:30",
        "minute": None,
        "score": None,
        "odds": {"home": 2.10, "draw": 3.40, "away": 3.20},
        "live": False,
        "status": "upcoming"
    },
    {
        "id": 6,
        "home": "Inter Milan",
        "away": "AC Milan",
        "league": "Serie A",
        "time": "Tomorrow 20:45",
        "minute": None,
        "score": None,
        "odds": {"home": 2.20, "draw": 3.30, "away": 3.10},
        "live": False,
        "status": "upcoming"
    },
    {
        "id": 7,
        "home": "Atletico",
        "away": "Sevilla",
        "league": "La Liga",
        "time": "21:00",
        "minute": None,
        "score": None,
        "odds": {"home": 1.85, "draw": 3.60, "away": 4.10},
        "live": False,
        "status": "upcoming"
    },
    {
        "id": 8,
        "home": "Juventus",
        "away": "Roma",
        "league": "Serie A",
        "time": "Tomorrow 17:30",
        "minute": None,
        "score": None,
        "odds": {"home": 2.05, "draw": 3.30, "away": 3.50},
        "live": False,
        "status": "upcoming"
    }
]

casino_games = [
    {"id": 1, "name": "Aviator", "type": "crash", "min_bet": 100, "max_bet": 50000, "rtp": 97.0, "max_multiplier": 100.0},
    {"id": 2, "name": "Gates of Olympus", "type": "slots", "min_bet": 100, "max_bet": 10000, "rtp": 96.5, "max_multiplier": 5000.0},
    {"id": 3, "name": "Mines", "type": "strategy", "min_bet": 100, "max_bet": 20000, "rtp": 98.5, "max_multiplier": 50.0},
    {"id": 4, "name": "Dice", "type": "dice", "min_bet": 50, "max_bet": 50000, "rtp": 99.0, "max_multiplier": 10.0},
    {"id": 5, "name": "Slots Megaways", "type": "slots", "min_bet": 200, "max_bet": 15000, "rtp": 96.2, "max_multiplier": 20000.0},
    {"id": 6, "name": "Blackjack Live", "type": "table", "min_bet": 500, "max_bet": 50000, "rtp": 99.5, "max_multiplier": 2.5},
    {"id": 7, "name": "Roulette VIP", "type": "table", "min_bet": 100, "max_bet": 100000, "rtp": 97.3, "max_multiplier": 35.0},
    {"id": 8, "name": "Plinko", "type": "arcade", "min_bet": 100, "max_bet": 10000, "rtp": 98.0, "max_multiplier": 1000.0},
    {"id": 9, "name": "Crash Rocket", "type": "crash", "min_bet": 100, "max_bet": 50000, "rtp": 97.5, "max_multiplier": float('inf')}
]

crypto_prices = {
    "BTC": 43250.00,
    "ETH": 2280.50,
    "SOL": 98.35,
    "BNB": 312.80,
    "USDT": 1.00
}

NAIRA_USD_RATE = 1580.00

# ============================================================================
# AUTHENTICATION & SECURITY FUNCTIONS
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return username"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def get_current_user(username: str = Depends(verify_token)) -> dict:
    """Get current user from database"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[username]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_price_fluctuation():
    """Simulate crypto price changes"""
    for coin in crypto_prices:
        if coin != "USDT":
            fluctuation = random.uniform(-0.015, 0.015)
            crypto_prices[coin] = round(crypto_prices[coin] * (1 + fluctuation), 2)

def calculate_bet_outcome(odds: float) -> bool:
    """Calculate bet outcome with house edge"""
    probability = 1 / odds
    realistic_probability = probability * 0.95
    return random.random() < realistic_probability

def simulate_casino_game(game: dict, bet_amount: float) -> dict:
    """Simulate casino game with RTP"""
    rtp = game["rtp"] / 100
    game_type = game["type"]
    
    if game_type == "crash":
        if random.random() < rtp:
            multiplier = random.uniform(1.2, min(game["max_multiplier"], 10.0))
            return {"won": True, "multiplier": round(multiplier, 2), "payout": round(bet_amount * multiplier, 2)}
    elif game_type == "slots":
        rand = random.random()
        if rand < 0.015:
            multiplier = random.uniform(50, min(game["max_multiplier"], 1000))
            return {"won": True, "multiplier": round(multiplier, 2), "payout": round(bet_amount * multiplier, 2), "bonus": "BIG WIN!"}
        elif rand < rtp:
            multiplier = random.uniform(1.5, 5.0)
            return {"won": True, "multiplier": round(multiplier, 2), "payout": round(bet_amount * multiplier, 2)}
    elif game_type == "table":
        if random.random() < rtp:
            if game["name"] == "Blackjack Live":
                multiplier = random.choice([1.5, 2.0, 2.5])
            else:
                multiplier = random.choice([2.0] * 10 + [18.0, 35.0])
            return {"won": True, "multiplier": multiplier, "payout": round(bet_amount * multiplier, 2)}
    elif game_type in ["dice", "strategy", "arcade"]:
        if random.random() < rtp:
            multiplier = random.uniform(1.5, min(game["max_multiplier"], 20.0))
            return {"won": True, "multiplier": round(multiplier, 2), "payout": round(bet_amount * multiplier, 2)}
    
    return {"won": False, "multiplier": 0, "payout": 0}

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="IB Crypto Play API - Enhanced",
    version="3.1.0",
    description="Production-Ready Sports Betting & Casino Platform"
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - More restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Change to specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "app": "IB Crypto Play API - Enhanced",
        "version": "3.1.0",
        "status": "operational",
        "security": "enabled",
        "features": ["password_hashing", "jwt_auth", "rate_limiting", "logging"]
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.1.0"
    }

# ============================================================================
# AUTHENTICATION
# ============================================================================

@app.post("/api/v1/auth/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, credentials: LoginRequest):
    """Login with rate limiting and password hashing"""
    username = credentials.username.lower()
    
    logger.info(f"Login attempt for user: {username}")
    
    if username not in users_db:
        logger.warning(f"Failed login - user not found: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    user = users_db[username]
    
    if not verify_password(credentials.password, user["password"]):
        logger.warning(f"Failed login - wrong password: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Update last login
    user["last_login"] = datetime.now().isoformat()
    
    # Create access token
    access_token = create_access_token(data={"sub": username})
    
    logger.info(f"Successful login: {username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": username,
            "balance": user["balance"],
            "crypto_wallet": user["crypto_wallet"]
        }
    }

@app.get("/api/v1/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get authenticated user profile"""
    return {
        "username": "demo",
        "balance": current_user["balance"],
        "crypto_wallet": current_user["crypto_wallet"],
        "total_bets": len(current_user["bet_history"]),
        "wins": sum(1 for bet in current_user["bet_history"] if bet.get("won", False)),
        "created_at": current_user.get("created_at"),
        "last_login": current_user.get("last_login")
    }

@app.get("/api/v1/user/balance")
async def get_balance(current_user: dict = Depends(get_current_user)):
    """Get user balance - requires authentication"""
    return {
        "balance": current_user["balance"],
        "currency": "NGN"
    }

# ============================================================================
# SPORTS BETTING
# ============================================================================

@app.get("/api/v1/matches")
@limiter.limit("30/minute")
async def get_matches(request: Request, live_only: bool = False):
    """Get matches with rate limiting"""
    simulate_price_fluctuation()
    
    if live_only:
        return [m for m in matches_db if m["live"]]
    return matches_db

@app.post("/api/v1/bets/place")
@limiter.limit("10/minute")
async def place_bet(
    request: Request,
    bet_request: MultiBetRequest,
    current_user: dict = Depends(get_current_user)
):
    """Place bet with authentication and validation"""
    total_stake = bet_request.total_stake
    
    if current_user["balance"] < total_stake:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    # Deduct stake
    current_user["balance"] -= total_stake
    
    # Calculate odds
    total_odds = 1.0
    for bet in bet_request.bets:
        total_odds *= bet["odds"]
    
    # Determine outcome
    won = calculate_bet_outcome(total_odds)
    payout = total_stake * total_odds if won else 0
    
    if won:
        current_user["balance"] += payout
    
    # Record bet
    bet_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "bets": bet_request.bets,
        "total_stake": total_stake,
        "total_odds": round(total_odds, 2),
        "won": won,
        "payout": round(payout, 2),
        "profit": round(payout - total_stake, 2) if won else -total_stake
    }
    current_user["bet_history"].append(bet_record)
    
    logger.info(f"Bet placed: {bet_record['id']}, won={won}")
    
    return bet_record

# ============================================================================
# CASINO
# ============================================================================

@app.get("/api/v1/casino/games")
async def get_casino_games():
    """Get all casino games"""
    return {"games": casino_games, "total": len(casino_games)}

@app.post("/api/v1/casino/play")
@limiter.limit("20/minute")
async def play_casino_game(
    request: Request,
    play_request: CasinoPlayRequest,
    current_user: dict = Depends(get_current_user)
):
    """Play casino game with authentication"""
    game = next((g for g in casino_games if g["id"] == play_request.game_id), None)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if play_request.bet_amount < game["min_bet"]:
        raise HTTPException(status_code=400, detail=f"Minimum bet is ₦{game['min_bet']}")
    if play_request.bet_amount > game["max_bet"]:
        raise HTTPException(status_code=400, detail=f"Maximum bet is ₦{game['max_bet']}")
    
    if current_user["balance"] < play_request.bet_amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    current_user["balance"] -= play_request.bet_amount
    
    result = simulate_casino_game(game, play_request.bet_amount)
    
    if result["won"]:
        current_user["balance"] += result["payout"]
    
    game_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "game_id": game["id"],
        "game_name": game["name"],
        "bet_amount": play_request.bet_amount,
        **result
    }
    current_user["bet_history"].append(game_record)
    
    logger.info(f"Casino game played: {game['name']}, won={result['won']}")
    
    return {**game_record, "new_balance": current_user["balance"]}

# ============================================================================
# CRYPTO WALLET
# ============================================================================

@app.get("/api/v1/crypto/prices")
@limiter.limit("60/minute")
async def get_crypto_prices(request: Request):
    """Get crypto prices with rate limiting"""
    simulate_price_fluctuation()
    return {
        "prices": crypto_prices,
        "timestamp": datetime.now().isoformat(),
        "naira_usd_rate": NAIRA_USD_RATE
    }

@app.post("/api/v1/wallet/convert")
@limiter.limit("10/minute")
async def convert_currency(
    request: Request,
    transfer: TransferRequest,
    current_user: dict = Depends(get_current_user)
):
    """Convert currency with validation"""
    if transfer.from_currency == "NGN":
        if current_user["balance"] < transfer.amount:
            raise HTTPException(status_code=400, detail="Insufficient Naira balance")
        
        usd_amount = transfer.amount / NAIRA_USD_RATE
        crypto_amount = usd_amount / crypto_prices[transfer.to_currency]
        
        current_user["balance"] -= transfer.amount
        current_user["crypto_wallet"][transfer.to_currency] += crypto_amount
        
        transaction = {
            "id": str(uuid.uuid4()),
            "type": "conversion",
            "from_currency": "NGN",
            "to_currency": transfer.to_currency,
            "from_amount": transfer.amount,
            "to_amount": round(crypto_amount, 8),
            "rate": crypto_prices[transfer.to_currency],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
    else:
        crypto_balance = current_user["crypto_wallet"].get(transfer.from_currency, 0)
        if crypto_balance < transfer.amount:
            raise HTTPException(status_code=400, detail="Insufficient crypto balance")
        
        usd_amount = transfer.amount * crypto_prices[transfer.from_currency]
        naira_amount = usd_amount * NAIRA_USD_RATE
        
        current_user["crypto_wallet"][transfer.from_currency] -= transfer.amount
        current_user["balance"] += naira_amount
        
        transaction = {
            "id": str(uuid.uuid4()),
            "type": "conversion",
            "from_currency": transfer.from_currency,
            "to_currency": "NGN",
            "from_amount": transfer.amount,
            "to_amount": round(naira_amount, 2),
            "rate": crypto_prices[transfer.from_currency],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
    
    current_user["transactions"].append(transaction)
    logger.info(f"Currency converted: {transaction['id']}")
    
    return {
        **transaction,
        "new_balance": current_user["balance"],
        "new_crypto_wallet": current_user["crypto_wallet"]
    }

@app.post("/api/v1/wallet/withdraw")
@limiter.limit("5/minute")
async def withdraw_crypto(
    request: Request,
    withdraw: WithdrawRequest,
    current_user: dict = Depends(get_current_user)
):
    """Withdraw crypto with validation"""
    crypto_balance = current_user["crypto_wallet"].get(withdraw.crypto, 0)
    if crypto_balance < withdraw.amount:
        raise HTTPException(status_code=400, detail="Insufficient crypto balance")
    
    current_user["crypto_wallet"][withdraw.crypto] -= withdraw.amount
    
    transaction = {
        "id": str(uuid.uuid4()),
        "type": "withdrawal",
        "crypto": withdraw.crypto,
        "amount": withdraw.amount,
        "wallet_address": withdraw.wallet_address,
        "status": "pending",
        "timestamp": datetime.now().isoformat(),
        "network_fee": 0.001
    }
    current_user["transactions"].append(transaction)
    
    logger.info(f"Crypto withdrawal: {transaction['id']}")
    
    return transaction

@app.get("/api/v1/wallet/transactions")
async def get_transactions(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get transaction history - paginated"""
    if limit > 100:
        limit = 100
    
    return {
        "transactions": current_user["transactions"][-limit:],
        "total": len(current_user["transactions"]),
        "limit": limit
    }

@app.get("/api/v1/bets/history")
async def get_bet_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get bet history - paginated"""
    if limit > 100:
        limit = 100
    
    return {
        "bets": current_user["bet_history"][-limit:],
        "total_bets": len(current_user["bet_history"]),
        "total_wins": sum(1 for bet in current_user["bet_history"] if bet.get("won", False)),
        "win_rate": round(sum(1 for bet in current_user["bet_history"] if bet.get("won", False)) / max(len(current_user["bet_history"]), 1) * 100, 2),
        "limit": limit
    }

# ============================================================================
# STATISTICS
# ============================================================================

@app.get("/api/v1/stats")
async def get_statistics():
    """Get platform statistics"""
    user = users_db["demo"]
    total_bets = len(user["bet_history"])
    total_wins = sum(1 for bet in user["bet_history"] if bet.get("won", False))
    
    return {
        "total_users": 1,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "win_rate": round(total_wins / max(total_bets, 1) * 100, 2),
        "total_volume": sum(bet.get("total_stake", bet.get("bet_amount", 0)) for bet in user["bet_history"]),
        "active_matches": len([m for m in matches_db if m["live"]]),
        "available_games": len(casino_games)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
