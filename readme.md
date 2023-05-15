# DataCraft

Proje, yapay zeka algoritmalarını kullanarak farklı problemleri çözmek için geliştirilmiş bir yazılımdır.
Bu projeyi yapma amacım yapay zeka algoritmalarının mantığını anlamak ve kendimi Python 'da geliştirmektir.

## Ana Proje 

Bu sürüm, el yazısı rakamların üretilmesi için bir GAN modeli kullanır. Model, üreteç ve diskriminatör olmak üzere iki aşamadan oluşur. Üreteç, rastgele gürültüyü alır ve gerçekçi el yazısı rakamları üretir. Diskriminatör ise, gerçek ve üretilmiş rakamları ayırt etmek için eğitilir.

Kodları içeren dosya: `main.py`

Özellikler:
- El yazısı rakamların üretilmesi için GAN modelini kullanır.
- MNIST veri setini kullanarak eğitim yapar.
- Üreteç ve diskriminatör ağlarını tanımlar ve eğitir.
- Üretilen rakamları ve eğitim ilerlemesini görselleştirir.

Kullanım:
1. Gerekli kütüphaneleri yükleyin: TensorFlow, Keras, NumPy, Pandas
2. Veri setini yükleyin (ör. MNIST veri seti).
3. Modeli eğitmek için kodu çalıştırın.
4. Üretilen rakamları ve eğitim ilerlemesini gözlemleyin.
5. İsteğe bağlı olarak, modeli kaydedebilir ve sonradan kullanabilirsiniz.

Notlar:
- Modeli daha uzun süre eğiterek daha iyi sonuçlar alabilirsiniz.
- GAN'lerin eğitimi genellikle kararlı olmayabilir, deneme yanılma süreci gerektirebilir.
- GAN modelleriyle oynamak ve değişiklikler yapmak cesaret gerektirebilir, sonuçlar değişebilir.

Örnek Çıktı:
![El Yazısı Rakam Örneği](example_output.png)

Daha fazla bilgi ve detaylar için lütfen kod dosyasını inceleyin ve kaynak belirtmeyi unutmayın.

## 1. Sürüm

Bu sürüm, metin sınıflandırması yapmak için bir destek vektör makinesi (SVM) kullanır. Veri seti üzerindeki metinleri vektörlere dönüştürür ve SVM modelini eğitir. Ardından, doğruluk oranını hesaplar.

Kodları içeren dosya: `relase1.py`

Özellikler:
- Metin sınıflandırması için SVM algoritması kullanır.
- Veri setini önişler ve vektörlere dönüştürür.
- Eğitim ve test veri setlerini ayırır.
- Doğruluk oranını hesaplar ve ekrana yazdırır.

## 2. Sürüm

Bu sürüm, zaman serisi tahmini yapmak için bir uzun-kısa süreli hafıza (LSTM) modeli kullanır. Veri setini ölçeklendirir, kategorik etiketlere dönüştürür ve ardından LSTM modelini eğitir. Son olarak, doğruluk oranını hesaplar.

Kodları içeren dosya: `relase2.py`

Özellikler:
- Zaman serisi tahmini için LSTM modelini kullanır.
- Veri setini ölçeklendirir ve kategorik etiketlere dönüştürür.
- Eğitim ve test veri setlerini ayırır.
- Doğruluk oranını hesaplar ve ekrana yazdırır.

## 3. Sürüm

Bu sürüm, görüntü sınıflandırması yapmak için bir evrişimli sinir ağı (CNN) kullanır. Evrişimli ve tam bağlantılı katmanlardan oluşan bir modeli eğitir. Veri setini girişe uygun hale getirir, eğitim ve test veri setlerini oluşturur ve modeli eğitir. Son olarak, kayıp ve doğruluk oranlarını hesaplar.

Kodları içeren dosya: `relase3.py`

Özellikler:
- Görüntü sınıflandırması için evrişimli sinir ağı (CNN) kullanır.
- Veri setini eğitim ve test veri setlerine ayırır.
- Veri setini ön işler, boyutlandırır ve normalleştirir.
- Evrişimli ve tam bağlantılı katmanlardan oluşan bir modeli eğitir.
- Kayıp ve doğruluk oranlarını hesaplar ve ekrana yazdırır.
