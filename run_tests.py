import unittest
import sys
import os
from colorama import init, Fore, Style, Back

def run_tests():
    """Tüm testleri çalıştırır ve sonuçları raporlar."""
    # Colorama'yı başlat
    init()

    # Proje kök dizinini ekle
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test dizinini ekle
    test_dir = os.path.join(project_root, "tests")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # Test paketlerini keşfet
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")
    
    # Test sonuçları için özel runner
    class ColorTextTestResult(unittest.TextTestResult):
        def startTest(self, test):
            self.stream.write(f"\n{Fore.CYAN}{test.shortDescription() or str(test)}{Style.RESET_ALL}\n")
            super().startTest(test)

        def addSuccess(self, test):
            self.stream.write(f"{Fore.GREEN}✓ OK{Style.RESET_ALL}\n")
            super().addSuccess(test)

        def addError(self, test, err):
            self.stream.write(f"{Back.RED}{Fore.WHITE} HATA {Style.RESET_ALL}\n")
            self.stream.write(f"{Fore.RED}{self._exc_info_to_string(err, test)}{Style.RESET_ALL}\n")
            super().addError(test, err)

        def addFailure(self, test, err):
            self.stream.write(f"{Back.RED}{Fore.WHITE} BAŞARISIZ {Style.RESET_ALL}\n")
            self.stream.write(f"{Fore.RED}{self._exc_info_to_string(err, test)}{Style.RESET_ALL}\n")
            super().addFailure(test, err)

        def addSkip(self, test, reason):
            self.stream.write(f"{Fore.YELLOW}⚠ ATLANDI{Style.RESET_ALL} ({reason})\n")
            super().addSkip(test, reason)

        def addExpectedFailure(self, test, err):
            self.stream.write(f"{Fore.YELLOW}✓ BEKLENEN BAŞARISIZLIK{Style.RESET_ALL}\n")
            super().addExpectedFailure(test, err)

        def addUnexpectedSuccess(self, test):
            self.stream.write(f"{Fore.RED}⚠ BEKLENMEDİK BAŞARI{Style.RESET_ALL}\n")
            super().addUnexpectedSuccess(test)

    class TestRunner(unittest.TextTestRunner):
        def __init__(self):
            super().__init__(verbosity=2, resultclass=ColorTextTestResult)
            self.results = {}
            self.failures_details = []
            self.errors_details = []
        
        def run(self, test):
            result = super().run(test)
            self.results = {
                "total": result.testsRun,
                "passed": result.testsRun - len(result.failures) - len(result.errors),
                "failed": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped)
            }
            
            # Hata detaylarını kaydet
            for failure in result.failures:
                self.failures_details.append({
                    "test": str(failure[0]),
                    "message": str(failure[1])
                })
            
            for error in result.errors:
                self.errors_details.append({
                    "test": str(error[0]),
                    "message": str(error[1])
                })
            
            return result
    
    # Testleri çalıştır
    runner = TestRunner()
    result = runner.run(suite)
    
    # Sonuçları raporla
    print("\n" + "=" * 70)
    print(f"{Back.WHITE}{Fore.BLACK}{Style.BRIGHT} Test Sonuçları {Style.RESET_ALL}")
    print("=" * 70)
    
    # Genel özet
    print(f"\n{Style.BRIGHT}Özet:{Style.RESET_ALL}")
    print(f"Toplam Test: {Style.BRIGHT}{runner.results['total']}{Style.RESET_ALL}")
    print(f"Başarılı: {Fore.GREEN}{Style.BRIGHT}{runner.results['passed']}{Style.RESET_ALL}")
    if runner.results['failed'] > 0:
        print(f"Başarısız: {Back.RED}{Fore.WHITE}{Style.BRIGHT}{runner.results['failed']}{Style.RESET_ALL}")
    else:
        print(f"Başarısız: {Fore.GREEN}0{Style.RESET_ALL}")
    if runner.results['errors'] > 0:
        print(f"Hata: {Back.RED}{Fore.WHITE}{Style.BRIGHT}{runner.results['errors']}{Style.RESET_ALL}")
    else:
        print(f"Hata: {Fore.GREEN}0{Style.RESET_ALL}")
    if runner.results['skipped'] > 0:
        print(f"Atlanmış: {Fore.YELLOW}{Style.BRIGHT}{runner.results['skipped']}{Style.RESET_ALL}")
    else:
        print(f"Atlanmış: 0")
    
    # Başarısız testlerin detayları
    if runner.failures_details:
        print(f"\n{Back.RED}{Fore.WHITE}{Style.BRIGHT} Başarısız Test Detayları {Style.RESET_ALL}")
        for i, failure in enumerate(runner.failures_details, 1):
            print(f"\n{Fore.RED}{Style.BRIGHT}Başarısız Test #{i}:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Test:{Style.RESET_ALL} {failure['test']}")
            print(f"{Fore.YELLOW}Hata Mesajı:{Style.RESET_ALL}\n{failure['message']}")
    
    # Hata veren testlerin detayları
    if runner.errors_details:
        print(f"\n{Back.RED}{Fore.WHITE}{Style.BRIGHT} Hata Detayları {Style.RESET_ALL}")
        for i, error in enumerate(runner.errors_details, 1):
            print(f"\n{Fore.RED}{Style.BRIGHT}Hata #{i}:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Test:{Style.RESET_ALL} {error['test']}")
            print(f"{Fore.YELLOW}Hata Mesajı:{Style.RESET_ALL}\n{error['message']}")
    
    print("\n" + "=" * 70)
    
    # Test sonucuna göre renklendirme
    if result.wasSuccessful():
        print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} BAŞARILI {Style.RESET_ALL} Tüm testler başarıyla tamamlandı.")
    else:
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT} BAŞARISIZ {Style.RESET_ALL} Bazı testler başarısız oldu.")
    
    print("=" * 70 + "\n")
    
    # Başarı durumunu döndür
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 