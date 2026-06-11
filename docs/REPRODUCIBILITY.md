# Yeniden Üretim Notları

## Ortam

- Python 3.9+
- CUDA uyumlu PyTorch kurulumu (eğitim ve büyük modeller için)
- FFmpeg
- Kök `requirements.txt` içindeki Python bağımlılıkları

Model ağırlıkları ve ses kayıtları GitHub deposunda tutulmaz. Yayımlanmış M1/M2
modelleri ve medv3 verisi Hugging Face'ten indirilmelidir. Genel benchmark sesleri
ise ilgili kaynak veri kümelerinin kullanım koşullarına göre edinilmelidir.

Ham veri denetleme ve temizleme betikleri varsayılan olarak `data/raw` altını
kullanır. Farklı konumlar aşağıdaki ortam değişkenleriyle verilebilir:

```powershell
$env:TURKMED_CV_DIR = "C:\data\commonvoice_tr"
$env:TURKMED_ISSAI_DIR = "C:\data\issai"
$env:TURKMED_OPENSLR_DIR = "C:\data\openslr_tr"
```

## Manifest Biçimi

Ana değerlendirme araçları en az aşağıdaki alanları içeren bir CSV manifesti
bekler:

```csv
audio_path,reference_text,source,split
C:/data/example.wav,örnek referans metni,example,test
```

`audio_path` erişilebilir bir ses dosyasını göstermelidir. Manifest doğrulaması:

```powershell
python -c "from turkmed_stt.manifest import validate_manifest; print(validate_manifest('manifest.csv'))"
```

## Tek Model Benchmarkı

```powershell
python -m turkmed_stt.benchmark
```

Programatik kullanım için `turkmed_stt.benchmark.run_benchmark` işlevine manifest,
çıktı CSV yolu, backend ve model kimliği verilir.

## Model Matrisi

```powershell
python scripts/run_benchmark_matrix.py `
  --manifest C:\data\benchmark_manifest.csv `
  --models configs\benchmark_models_20.csv `
  --output-dir runs\benchmark_matrix
```

Özet üretimi:

```powershell
python scripts/summarize_benchmark_matrix.py `
  --run-dir runs\benchmark_matrix
```

## LoRA Eğitimi ve Birleştirme

Eğitim betiğinin tüm seçenekleri:

```powershell
python scripts/train_lora_whisper_v2.py --help
```

M1 ve M2 adapterleri eğitim tamamlandıktan sonra temel modelle birleştirilir:

```powershell
python scripts/merge_lora.py ADAPTER_DIZINI CIKTI_DIZINI
```

M1/M2 için kullanılan temel ayarlar `results/finetune/*_training_metadata.json`
dosyalarında özetlenmiştir. Büyük ölçekli eğitimin birebir tekrarı için aynı veri
sürümleri, örnekleme düzeni, rastgelelik tohumları ve donanım ortamı
sabitlenmelidir.

## İstatistiksel Analiz

V2 gerçek konuşma sonuçlarının eşleştirilmiş bootstrap analizi:

```powershell
python scripts/analyze_v2_bootstrap.py --help
```

Yayımlanan özetler:

- `results/real-speech/V2_BOOTSTRAP_SIGNIFICANCE.md`
- `results/real-speech/v2_bootstrap_significance.json`

## Kapsam Sınırı

Bu depo deneylerin çalıştırılmasını sağlayan kodu ve özet sonuçları yayımlar;
lisans, boyut ve mahremiyet nedenleriyle ham ses arşivini içermez. Sonuçları
yeniden üretmek isteyen kullanıcı kendi yasal veri kopyalarını ve manifest
yollarını sağlamalıdır.
