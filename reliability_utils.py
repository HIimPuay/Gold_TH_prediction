#!/usr/bin/env python3
"""
reliability_utils.py - Scalability & Reliability utilities
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Health checks
- Error handling wrappers
"""
import time
import functools
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
import logging
from pathlib import Path
import json

# ==================== LOGGING SETUP ====================

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"reliability_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==================== RETRY MECHANISM ====================

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_retries: จำนวนครั้งที่ retry สูงสุด
        initial_delay: delay เริ่มต้น (วินาที)
        max_delay: delay สูงสุด (วินาที)
        exponential_base: base สำหรับ exponential backoff
        exceptions: tuple ของ exceptions ที่จะ retry
    
    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def fetch_data():
            # your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"⚠️  {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                    
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
        return wrapper
    return decorator


# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """
    Circuit Breaker pattern implementation
    
    States:
    - CLOSED: ทำงานปกติ
    - OPEN: หยุดการทำงาน (เกิด error เยอะ)
    - HALF_OPEN: ลองทำงานอีกครั้ง
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        @breaker
        def risky_operation():
            # your code here
            pass
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"🔄 Circuit breaker HALF_OPEN for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exceptions as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """ตรวจสอบว่าควร reset หรือไม่"""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """เรียกเมื่อสำเร็จ"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("✅ Circuit breaker CLOSED")
    
    def _on_failure(self):
        """เรียกเมื่อล้มเหลว"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"🚨 Circuit breaker OPEN (failures: {self.failure_count})")


# ==================== HEALTH CHECK ====================

class HealthCheck:
    """
    ระบบตรวจสอบสุขภาพของ service
    
    Example:
        health = HealthCheck()
        health.add_check("database", check_database_connection)
        health.add_check("model", check_model_loaded)
        
        status = health.check_all()
    """
    
    def __init__(self):
        self.checks = {}
        self.history = []
    
    def add_check(self, name: str, check_func: Callable) -> None:
        """เพิ่ม health check"""
        self.checks[name] = check_func
    
    def check_all(self) -> dict:
        """รัน health checks ทั้งหมด"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results['checks'][name] = {
                    'status': 'healthy',
                    'details': result
                }
            except Exception as e:
                results['checks'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        # บันทึกประวัติ
        self.history.append(results)
        self.history = self.history[-100:]  # เก็บ 100 รายการล่าสุด
        
        return results
    
    def get_uptime_percentage(self, hours: int = 24) -> float:
        """คำนวณ uptime %"""
        if not self.history:
            return 100.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            h for h in self.history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_checks:
            return 100.0
        
        healthy_count = sum(1 for h in recent_checks if h['overall_status'] == 'healthy')
        return (healthy_count / len(recent_checks)) * 100


# ==================== ERROR HANDLER DECORATOR ====================

def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    raise_on_error: bool = False
):
    """
    Decorator สำหรับจัดการ errors
    
    Args:
        default_return: ค่าที่จะ return เมื่อเกิด error
        log_errors: บันทึก error log หรือไม่
        raise_on_error: raise exception หรือไม่
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"❌ Error in {func.__name__}: {e}", exc_info=True)
                
                if raise_on_error:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


# ==================== TIMEOUT DECORATOR ====================

def timeout(seconds: int):
    """
    Timeout decorator (Unix only)
    
    Example:
        @timeout(30)
        def long_running_task():
            # your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds} seconds")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel timeout
            
            return result
        
        return wrapper
    return decorator


# ==================== RATE LIMITER ====================

class RateLimiter:
    """
    Rate limiter สำหรับจำกัดจำนวน requests
    
    Example:
        limiter = RateLimiter(max_calls=100, period=60)
        
        @limiter
        def api_call():
            # your code here
            pass
    """
    
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            
            # ลบ calls ที่เก่าเกินไป
            self.calls = [c for c in self.calls if now - c < self.period]
            
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (now - self.calls[0])
                raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f} seconds")
            
            self.calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper


# ==================== EXAMPLE USAGE ====================

# Example: API call with retry and circuit breaker
api_breaker = CircuitBreaker(failure_threshold=3, timeout=60)

@retry_with_exponential_backoff(max_retries=3)
@api_breaker
@handle_errors(default_return=None)
def fetch_external_api():
    """ดึงข้อมูลจาก external API"""
    import requests
    response = requests.get("https://api.example.com/data", timeout=10)
    response.raise_for_status()
    return response.json()


# Example: Health checks
def check_model_loaded():
    """ตรวจสอบว่าโมเดลโหลดแล้วหรือยัง"""
    from pathlib import Path
    model_path = Path("model/best_model.pkl")
    if not model_path.exists():
        raise Exception("Model file not found")
    return {"model_size_mb": model_path.stat().st_size / (1024 * 1024)}


def check_feature_store():
    """ตรวจสอบ feature store"""
    from pathlib import Path
    fs_path = Path("data/Feature_store/feature_store.csv")
    if not fs_path.exists():
        raise Exception("Feature store not found")
    
    import pandas as pd
    df = pd.read_csv(fs_path)
    return {
        "rows": len(df),
        "latest_date": df['date'].max()
    }


# ==================== MAIN ====================

def main():
    """ตัวอย่างการใช้งาน"""
    print("\n" + "=" * 70)
    print("🛡️  RELIABILITY & SCALABILITY UTILITIES")
    print("=" * 70)
    
    # 1. ทดสอบ Retry mechanism
    print("\n1️⃣  Testing Retry Mechanism...")
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Random failure")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 2. ทดสอบ Circuit Breaker
    print("\n2️⃣  Testing Circuit Breaker...")
    
    breaker = CircuitBreaker(failure_threshold=2, timeout=5)
    
    @breaker
    def unreliable_service():
        raise Exception("Service unavailable")
    
    for i in range(5):
        try:
            unreliable_service()
        except Exception as e:
            print(f"   Attempt {i+1}: {str(e)[:50]}")
    
    # 3. ทดสอบ Health Checks
    print("\n3️⃣  Testing Health Checks...")
    
    health = HealthCheck()
    health.add_check("model", check_model_loaded)
    health.add_check("feature_store", check_feature_store)
    
    status = health.check_all()
    print(f"   Overall Status: {status['overall_status']}")
    
    for name, check in status['checks'].items():
        emoji = "✅" if check['status'] == 'healthy' else "❌"
        print(f"   {emoji} {name}: {check['status']}")
    
    # 4. ทดสอบ Rate Limiter
    print("\n4️⃣  Testing Rate Limiter...")
    
    limiter = RateLimiter(max_calls=3, period=5)
    
    @limiter
    def limited_api():
        return "API response"
    
    for i in range(5):
        try:
            result = limited_api()
            print(f"   Call {i+1}: {result}")
        except Exception as e:
            print(f"   Call {i+1}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()