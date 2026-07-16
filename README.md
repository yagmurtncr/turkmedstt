# TurkMedSTT

<p>
  <img src="https://github.com/yagmurtncr/turkmedstt/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Model-Whisper%20Large%20V3-EE4C2C?logo=pytorch&logoColor=white" alt="Whisper Large V3" />
  <a href="https://huggingface.co/turkmedstt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-turkmedstt-FFD21E" alt="Hugging Face" /></a>
</p>

> **Turkish general & medical automatic speech recognition (ASR).** Two Whisper Large V3
> models fine-tuned with LoRA, a 20-model benchmark under a shared evaluation protocol,
> an AcoSemantic semantic-preservation metric, and an ASR readability post-processor.
> *Full documentation below is in Turkish (graduation thesis).*

TurkMedSTT, Türkçe otomatik konuşma tanıma sistemlerini genel ve tıbbi alanlarda
inceleyen bir bitirme projesidir. Projede Whisper Large V3 tabanlı iki model
geliştirilmiş, 20 açık ASR modeli ortak bir değerlendirme protokolüyle
karşılaştırılmış ve sözcük/karakter hatalarına ek olarak anlamsal korunumu
inceleyen AcoSemantic değerlendirmesi uygulanmıştır.

## Proje Ekibi

- Muhammed Kumcu - [@muhammedkumcu](https://github.com/muhammedkumcu)
- Nur Yağmur Tuncer - [@yagmurtncr](https://github.com/yagmurtncr)

Danışman: Doç. Dr. Ayşe Berna Altınel Girgin

## Başlıca Çıktılar

- **M1 Genel Türkçe modeli:** Genel Türkçe verileriyle LoRA ince ayarı
- **M2 Tıbbi Türkçe modeli:** M1 üzerine genel ve tıbbi verilerle ikinci aşama LoRA
  ince ayarı
- **medv3 veri kümesi:** 3.236 Türkçe sentetik tıbbi konuşma kaydı
- **Genel Türkçe benchmark:** 20 model, 1.060 klip ve 21.200 model-klip sonucu
- **AcoSemantic değerlendirmesi:** ASR çıktılarında anlamsal ve duygusal korunumu
  inceleyen tamamlayıcı metrikler
- **ASR okunabilirlik post-processor:** ASR sonrasında casing, noktalama ve
  güvenilir kelime düzeltmeleri uygulayan V1 General ve V2 Medical modelleri
- **Demo ve leaderboard:** Modellerin denenebildiği ve sonuçların filtrelenebildiği
  Hugging Face Space uygulamaları

## Öne Çıkan Sonuçlar

Bağımsız 320 kliplik genel Türkçe değerlendirmesinde M1, temel modele göre WER'i
0,1213'ten 0,0792'ye, CER'i 0,0546'dan 0,0226'ya düşürmüştür. Bu değerler sırasıyla
%34,7 WER ve %58,6 CER göreli iyileşmesine karşılık gelir. M2 bu genel dil
kazanımlarını korumuştur.

Eğitim cümlelerinden farklı 516 gerçek konuşma kaydından oluşan zor tıbbi terim
testinde M2, üç konuşmacının tamamında temel modelden daha düşük WER üretmiştir.
Birleştirilmiş eşleştirilmiş bootstrap analizinde M0-M2 WER farkı 0,0203,
%95 güven aralığı [0,0122; 0,0284] ve p<0,0001'dir.

Ayrıntılı tablolar ve değerlendirme sınırları için [Sonuçlar](docs/RESULTS.md)
belgesine bakınız.

## Hugging Face Yayınları

- [TurkMedSTT organizasyonu](https://huggingface.co/turkmedstt)
- [M1 - Genel Türkçe model](https://huggingface.co/turkmedstt/whisper-large-v3-turkish-general)
- [M2 - Tıbbi Türkçe model](https://huggingface.co/turkmedstt/whisper-large-v3-turkish-medical)
- [medv3 tıbbi veri kümesi](https://huggingface.co/datasets/turkmedstt/medv3-turkish-medical-asr)
- [Genel Türkçe benchmark verileri](https://huggingface.co/datasets/turkmedstt/turkish-asr-benchmark)
- [ASR demo](https://huggingface.co/spaces/turkmedstt/turkmedstt-demo)
- [İnteraktif leaderboard](https://huggingface.co/spaces/turkmedstt/turkish-asr-leaderboard)
- [V1 Genel Türkçe post-processor](https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1)
- [V2 Sağlık post-processor](https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2)

Model ağırlıkları, ses kayıtları ve yayımlanmış veri dosyaları GitHub deposunu
gereksiz büyütmemek için burada tekrar tutulmamaktadır. Bunlara yukarıdaki
Hugging Face bağlantılarından erişilebilir.

## Depo Yapısı

```text
apps/                 Hugging Face demo ve leaderboard uygulamaları
configs/              Nihai benchmark model listesi
docs/figures/         Sistem diyagramları ve temel sonuç grafikleri
docs/thesis/          Nihai bitirme raporu (DOCX ve PDF)
docs/presentation/    Proje sunumu (PPTX)
results/              Benchmark, ince ayar ve gerçek konuşma sonuç özetleri
scripts/              Veri hazırlama, eğitim, değerlendirme ve yayın betikleri
postprocessing/       Türkçe ASR okunabilirlik son-işleme modeli ve sonuçları
static/               Yerel FastAPI arayüzünün statik dosyaları
turkmed_stt/          Ana Python paketi
```

## Okunabilirlik Son-İşleme

`postprocessing/` modülü, ham ASR çıktısını serbestçe yeniden yazmadan daha
okunabilir hale getiren tek-checkpoint çok görevli token editörünü içerir.

Genel Türkçe V1, test WER değerini `0.06179` seviyesinden `0.04277` seviyesine
düşürmüştür. Sağlık alanı V2 kontrollü sağlık metni bozulmalarında WER değerini
`0.31106` seviyesinden `0.18352` seviyesine düşürmüştür. V2 henüz gerçek sağlık
ASR çıktıları üzerinde doğrulanmamıştır.

Ayrıntılar için [post-processing README](postprocessing/README.md) ve
[nihai sonuçlar](postprocessing/reports/RESULTS.md) dosyalarına bakınız.

## Yerel Kurulum

Python 3.9 veya daha yeni bir sürüm önerilir. GPU ile değerlendirme ve eğitim
için CUDA uyumlu PyTorch kurulumu gereklidir.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Yerel FastAPI uygulaması:

```powershell
uvicorn turkmed_stt.app:app --reload
```

Hugging Face demo uygulaması:

```powershell
pip install -r apps/demo/requirements.txt
python apps/demo/app.py
```

Benchmark betikleri yerel ses dosyalarını içeren bir manifest bekler. Sesler bu
depoda bulunmadığından manifest yollarının kullanıcının veri konumuna göre
ayarlanması gerekir. Komut örnekleri ve yöntem notları
[Yeniden Üretim](docs/REPRODUCIBILITY.md) belgesindedir.

## Depoya Dahil Edilmeyenler

- Ham ve işlenmiş ses kayıtları
- Whisper temel modeli, LoRA adapterleri ve birleştirilmiş model ağırlıkları
- Kişisel bilgiler, erişim anahtarları ve yerel makineye özgü dosyalar
- Ara raporlar, geçici deney çıktıları, önbellekler ve yinelenen belgeler

## Etik ve Kullanım Sınırı

Bu çalışma araştırma ve eğitim amaçlıdır. Modeller tıbbi cihaz değildir; klinik
karar verme, tanı veya tedavi amacıyla doğrulanmadan kullanılmamalıdır. ASR
çıktıları özellikle ilaç adı, doz, sayı ve özel tıbbi terimler bakımından insan
tarafından kontrol edilmelidir.

## Lisans ve Atıf

Kaynak kod MIT License ile sunulmaktadır. Model ve veri kümelerinin
lisansları ilgili Hugging Face sayfalarında belirtilmiştir. Akademik kullanımda
[`CITATION.cff`](CITATION.cff) dosyasındaki bilgileri kullanınız.

Bu birleşik depo, temel TurkMedSTT çalışmasını
[`muhammedkumcu/turkmedstt`](https://github.com/muhammedkumcu/turkmedstt)
deposundan koruyarak okunabilirlik post-processing modülüyle genişletir.
