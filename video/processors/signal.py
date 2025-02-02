from typing import Callable, List

class Signal:
    """Basit bir sinyal/slot mekanizması."""
    
    def __init__(self):
        self._callbacks: List[Callable] = []
    
    def connect(self, callback: Callable) -> None:
        """Sinyale bir callback bağlar."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def disconnect(self, callback: Callable) -> None:
        """Sinyalden bir callback'i kaldırır."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def emit(self, *args, **kwargs) -> None:
        """Sinyali tetikler ve tüm callback'leri çağırır."""
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Signal callback error: {str(e)}")
    
    def clear(self) -> None:
        """Tüm callback'leri temizler."""
        self._callbacks.clear() 