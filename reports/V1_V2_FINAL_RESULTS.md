# Türkçe ASR Okunabilirlik Son-İşleme Modeli: Nihai Teknik Rapor

## 1. Yönetici Özeti

Bu projenin amacı, özellikle küçük ve yerel ortamda çalıştırılan ASR
modellerinin ürettiği Türkçe transkriptleri doğrudan kullanıcıya sunmadan önce
daha okunabilir hale getiren tek bir metin son-işleme modeli geliştirmektir.

Hedef, sesi yeniden transkribe etmek veya ASR modelinin yerine geçmek değildir.
Modelin görevi mevcut ASR çıktısı üzerinde güvenli ve sınırlı değişiklikler
yapmaktır:

```text
ham ASR çıktısı -> daha okunabilir ve mümkün olduğunca anlamı korunmuş metin
```

Çalışma sonunda iki ayrı ürün adayı oluşturulmuştur:

1. **V1 General:** Genel Türkçe ASR çıktıları için kabul edilen model.
2. **V2 Medical Synthetic:** Sağlık alanına uyarlanmış, ancak yalnızca sentetik
   sağlık metni bozulmaları üzerinde doğrulanmış domain modeli.

Genel kullanım için en başarılı ve güvenli model **V1 General** olmuştur. Kör
testte ortalama WER değerini `0.06179` seviyesinden `0.04277` seviyesine,
CER değerini ise `0.00940` seviyesinden `0.00631` seviyesine düşürmüştür.
Model 222 test satırını iyileştirmiş, 5 satırı kötüleştirmiş ve zaten doğru
olan hiçbir girdiyi bozmamıştır.

V2 Medical Synthetic, sentetik sağlık testinde güçlü sonuç vermiştir:
WER `0.31106 -> 0.18352`; 127 satır iyileşmiş ve hiçbir satır kötüleşmemiştir.
Ancak manifestlerde belirtilen sağlık WAV dosyaları diskte bulunmadığı için bu
model henüz gerçek sağlık ASR çıktıları üzerinde doğrulanmış değildir.

## 2. Problem Tanımı ve Projenin Amacı

Türkçe ASR çıktılarında yalnızca yanlış tanınmış kelimeler değil, aşağıdaki
okunabilirlik sorunları da sık görülmektedir:

- cümle başında küçük harf kullanımı;
- eksik veya yanlış noktalama;
- Türkçe karakterlerin kaybolması;
- yazım ve ek hataları;
- konuşma dilinden veya model hatasından kaynaklanan bozuk kelimeler;
- özel isimler ve alan terimlerinde yanlış tanıma.

Bu hataların tamamını serbest üretim yapan bir dil modeliyle düzeltmek risklidir.
Model metni daha akıcı hale getirirken ASR tarafından söylenmemiş kelimeler
ekleyebilir, mevcut içeriği silebilir veya anlamı değiştirebilir. Bu nedenle
proje boyunca temel ürün ilkesi şu olmuştur:

> Düzeltme kapsamını artırırken yanlış değişiklik ve halüsinasyon riskini
> sınırlamak.

Bu doğrultuda generative seq2seq yaklaşımı yerine çoğunlukla girdiyi kopyalayan,
yalnızca güveni yüksek token seviyesindeki aksiyonları uygulayan tek-checkpoint
çok görevli bir Transformer mimarisi tercih edilmiştir.

## 3. Veri Setinin Oluşturulması

### 3.1 Genel Türkçe Okunabilirlik Verisi

Temel okunabilirlik veri seti toplam `46.253` satırdan oluşmuştur:

| Veri türü | Satır |
| --- | ---: |
| Gerçek ASR kaynaklı okunabilirlik projeksiyonları | 31.918 |
| İçerik koruyan chat-normalization örnekleri | 14.335 |
| Toplam | 46.253 |

Veri kaynakları arasında Common Voice, MediaSpeech-TR ve çeşitli Türkçe metin
kaynakları bulunmaktadır. Eğitim verisi daha sonra kontrollü augmentasyonla
`110.384` token-edit örneğine genişletilmiştir.

Kullanılan split yapısı:

| Split | Satır |
| --- | ---: |
| Train | 102.640 |
| Validation | 3.828 |
| Test | 3.916 |

Nihai ASR odaklı değerlendirmede yalnızca
`asr_readability_projection` görev tipindeki validation ve test satırları
kullanılmıştır. Kör test kapsamı `3.191` gerçek ASR okunabilirlik satırıdır.

### 3.2 Sağlık Verisi

Sağlık manifestleri:

- `data/medical_corpus_v3_manifest.csv`
- `data/medical_corpus_v3_text_manifest.csv`

Bu manifestlerde:

- `3.236` kayıt;
- `809` benzersiz tıbbi cümle;
- 20 klinik alan;
- 4 sentetik konuşmacı profili bulunmaktadır.

Ancak manifestlerde belirtilen WAV dosyaları çalışma ortamında mevcut değildir.
Bu nedenle sağlık verisinden gerçek ASR çıktısı üretilememiştir. Bunun yerine
809 benzersiz sağlık cümlesinden kontrollü sentetik bozulmalar üretilmiştir:

- küçük harfe dönüştürme;
- noktalama kaldırma;
- Türkçe karakterleri ASCII karşılıklarına dönüştürme;
- tıbbi terimlerde kontrollü yazım bozulması.

Aynı kaynak cümlenin farklı splitlere girmesini önlemek için split işlemi
`source_sentence_id` temelinde yapılmıştır. Sonuçta `3.236` sentetik çift
üretilmiş, token sayısı hizalanabilen `3.092` örnek eğitimde kullanılmıştır.

## 4. Model Mimarisi

Kabul edilen model, yaklaşık 11,6 milyon parametreli
`ytu-ce-cosmos/turkish-mini-bert-uncased` encoder üzerine kurulmuş tek bir
çok görevli token editörüdür.

Tek checkpoint içerisinde üç ayrı görev başlığı bulunur:

1. **Casing başlığı:** Harfin korunması, büyütülmesi veya küçültülmesi.
2. **Noktalama başlığı:** Token sonrasında noktalama aksiyonu.
3. **Lexical edit başlığı:** Güveni yüksek, train-only replacement haritasından
   kontrollü kelime değişimi.

Bu tasarımın temel avantajı, tüm düzeltmelerin tek model checkpointinde
çalışması ve serbest metin üretimi yapılmamasıdır. Böylece yanlış kelime
ekleme, içerik silme ve halüsinasyon riski sınırlandırılmıştır.

## 5. Deneysel Süreç

### 5.1 Generative mT5 Deneyleri

İlk aşamalarda mT5 tabanlı generative modeller denenmiştir. Bu modeller pilot
eğitim bütçesinde güvenilir kopyalama davranışı öğrenememiş; bazı örneklerde
kelime silme ve içerik değiştirme riski göstermiştir.

Sonuç olarak generative model yaklaşımı üretim için reddedilmiştir. Daha büyük
bir generative model kullanmanın veri eksikliğini çözmeyeceği ve güvenlik
riskini artıracağı değerlendirilmiştir.

### 5.2 Tek Sınıflı Unified Editor

İlk unified modelde casing, noktalama ve lexical aksiyonlar tek büyük sınıf
uzayında birleştirilmiştir. Bu model validation üzerinde bazı satırları
iyileştirse de çok sayıda yanlış değişiklik ve doğru girdi bozulması üretmiştir.
Bu nedenle mimari reddedilmiştir.

### 5.3 Multi-Head V2 Pilot

Aynı encoder üzerinde görevlerin ayrı başlıklara bölünmesi önemli iyileşme
sağlamıştır. Pilot model:

- test WER değerini `0.06179 -> 0.04795`;
- test CER değerini `0.00940 -> 0.00709`

seviyesine düşürmüş; 192 satırı iyileştirirken hiçbir satırı
kötüleştirmemiştir. Bu deney, tek model içinde ayrı görev başlıklarının doğru
yön olduğunu göstermiştir.

### 5.4 Tam Veri Multi-Head V3

V3 eğitiminde:

- `102.640` özgün train satırının tamamı temsil edilmiştir;
- değişiklikli ve değişikliksiz örnekler dengeli örneklenmiştir;
- epoch başına `163.010` örnek kullanılmıştır;
- toplam 3 epoch ve `30.567` optimizer adımı tamamlanmıştır;
- her epoch sonunda validation yapılmış;
- test yalnızca final model seçildikten sonra çalıştırılmıştır.

Validation loss değerleri:

| Epoch | Validation loss |
| --- | ---: |
| 1 | 0.24758 |
| 2 | 0.23145 |
| 3 | 0.23014 |

En iyi checkpoint üçüncü epoch sonunda oluşmuştur.

### 5.5 Threshold Optimizasyonu

Modelin her tahminini doğrudan uygulamak yerine casing, punctuation ve lexical
başlıkları için ayrı confidence thresholdları seçilmiştir. Threshold seçimi
yalnızca validation üzerinde yapılmış; test sonuçları seçim işleminde
kullanılmamıştır.

İki deployment profili değerlendirilmiştir:

| Profil | Case | Punctuation | Lexical | İyileşen | Kötüleşen | Doğru Girdi Bozulması | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dengeli | 0.75 | 0.90 | 0.95 | 222 | 5 | 0 | 0.04277 |
| Güvenli | 0.75 | 0.98 | 0.95 | 216 | 1 | 0 | 0.04639 |

Genel V1 için dengeli profil seçilmiştir. Daha kritik uygulamalarda yanlış
değişiklik riskini azaltmak amacıyla güvenli profil kullanılabilir.

### 5.6 Açık Aksiyonlu V1 Deneyi

Önceki noktalama tasarımında `NONE` etiketi hem mevcut işareti koruma hem de
işaret üretmeme anlamına gelebiliyordu. Bu belirsizliği çözmek amacıyla yeni
bir açık aksiyon veri seti oluşturulmuştur:

- `KEEP`;
- `REMOVE`;
- `SET_PERIOD`, `SET_COMMA`, `SET_QUESTION` ve diğer işaretler;
- `FIX_DIACRITIC`;
- `FIX_SPELLING_OR_SUFFIX`;
- `REPLACE_LEXICAL`.

Bu deneyde `110.384` satır ve `4.173` train-only replacement adayı
kullanılmıştır. Model 3 epoch ve `30.867` adım eğitilmiştir.

Model bazı görevlerde küçük kazanımlar sağlamış olsa da genel kör testte:

- WER: `0.06179 -> 0.04290`;
- iyileşen / kötüleşen: `225 / 6`;
- doğru girdi bozulması: `1`

sonucunu vermiştir. Kabul edilen V1'i geçemediği ve bir doğru girdiyi bozduğu
için genel sürüm olarak **reddedilmiştir**.

Bu karar, yalnızca daha fazla satırı değiştiren modeli değil, güvenlik ve
genelleme açısından daha iyi modeli seçme ilkesine dayanmaktadır.

## 6. Nihai Genel Türkçe V1 Sonuçları

Kabul edilen genel model, Multi-Head V3 modelinin dengeli deployment profilidir.

| Metrik | Ham ASR | V1 General | Değişim |
| --- | ---: | ---: | ---: |
| WER | 0.06179 | 0.04277 | yaklaşık `%30,8` göreli azalma |
| CER | 0.00940 | 0.00631 | yaklaşık `%32,9` göreli azalma |

Satır bazlı sonuçlar:

| Ölçüm | Sonuç |
| --- | ---: |
| İyileştirilen satır | 222 |
| Kötüleştirilen satır | 5 |
| Doğru girdiyi bozma | 0 |

Görev bazlı ayrıntılı precision, recall ve F1 sonuçları ana sonuç tablosundan
ayrılmıştır. Bu metrikler karşılaştırmalı teknik inceleme amacıyla
`reports/MULTITASK_V3_ANALYSIS.md` ve
`reports/multitask_token_editor_v3_analysis.json` dosyalarında korunmaktadır.

### 6.1 Neden V1 General En Başarılı Modeldir?

V1 General, her görevde en yüksek recall değerini veren model değildir. Buna
rağmen genel kullanım için en başarılı model olarak seçilmesinin nedenleri
şunlardır:

- Kör testte en düşük kabul edilebilir genel WER/CER değerlerinden birini verir.
- Zaten doğru olan hiçbir test girdisini bozmaz.
- Yaptığı lexical değişiklikler sınırlı ve konservatiftir.
- Tek checkpoint olarak çalışır; çalışma zamanında model zinciri gerektirmez.
- Dengeli ve güvenli kullanım profilleri arasında seçim yapılabilir.
- Açık aksiyonlu deneyden daha iyi güvenlik-genelleme dengesi sağlar.

### 6.2 V1 General Modelinin Gerçek Gücü

Modelin en güçlü olduğu alan casing düzeltmesidir. Model güvenli bir
okunabilirlik düzenleyicisi olarak cümle başı büyük harf, belirli noktalama
davranışları ve sınırlı yüksek güvenli lexical düzeltmeler yapabilir.

### 6.3 V1 General Modelinin Eksikleri

Model henüz kapsamlı bir yanlış kelime düzelticisi değildir:

- lexical ve noktalama düzeltme kapsamı konservatiftir;
- virgül, soru işareti ve daha seyrek noktalama sınıflarında kapsama sınırlıdır;
- ses bilgisinde kaybolan kelimeleri yalnızca metinden güvenilir biçimde geri
  getiremez;
- kişi, yer ve özel alan terimlerinde gerçek insan doğrulamalı veri eksiktir.

Dolayısıyla V1, ağır bozulmuş ASR çıktısını tamamen kurtarmak yerine güvenli ve
sınırlı okunabilirlik iyileştirmesi için uygundur.

## 7. Sağlık Alanı V2 Sonuçları

V2 Medical Synthetic, genel model encoderından başlatılmış ve sağlık alanı
sentetik çiftleriyle 5 epoch boyunca fine-tune edilmiştir.

Sağlık eğitim verisi:

| Özellik | Değer |
| --- | ---: |
| Sentetik çift | 3.236 |
| Hizalanmış token-editor satırı | 3.092 |
| Train-only replacement adayı | 662 |
| Epoch | 5 |
| Optimizer adımı | 870 |

### 7.1 Held-Out Sentetik Sağlık Testi

| Metrik | Sentetik bozuk giriş | V2 Medical | Değişim |
| --- | ---: | ---: | ---: |
| WER | 0.31106 | 0.18352 | yaklaşık `%41,0` göreli azalma |
| CER | 0.05491 | 0.03547 | yaklaşık `%35,4` göreli azalma |

Satır bazlı sonuçlar:

| Ölçüm | Sonuç |
| --- | ---: |
| Değiştirilen satır | 136 |
| İyileştirilen satır | 127 |
| Kötüleştirilen satır | 0 |
| Doğru girdiyi bozma | 0 |

Görev bazlı ayrıntılı precision, recall ve F1 sonuçları karşılaştırmalı teknik
inceleme amacıyla `reports/postprocessor_v2_medical_analysis.json` dosyasında
korunmaktadır.

Bu sonuçlar, domain uyarlamasının kontrollü sağlık bozulmaları üzerinde
özellikle noktalama ve Türkçe karakter düzeltme performansını belirgin biçimde
artırabildiğini göstermektedir.

### 7.2 Genel Türkçe Testinde V2

Domain fine-tune sonrasında model genel Türkçe kör testte de değerlendirilmiştir:

| Metrik | Ham ASR | V2 Medical |
| --- | ---: | ---: |
| WER | 0.06179 | 0.04328 |
| CER | 0.00940 | 0.00639 |
| İyileşen / kötüleşen | - | 217 / 5 |
| Doğru girdiyi bozma | - | 0 |

V2 genel Türkçe yeteneğini tamamen kaybetmemiştir; ancak genel testte kabul
edilen V1 General'dan daha iyi değildir. Bu nedenle V2 yalnızca sağlık-domain
adayı olarak tutulmuştur.

### 7.3 Sağlık V2 İçin Kritik Sınırlama

V2'nin sağlık sonuçları gerçek sağlık seslerinden üretilmiş ASR çıktıları
üzerinde ölçülmemiştir. Manifestlerdeki WAV dosyaları mevcut olmadığından:

- gerçek doktor-hasta konuşma hataları;
- konuşmacı aksanı ve telaffuz farkları;
- arka plan gürültüsü;
- tıbbi terimlerin gerçek ASR tarafından yanlış tanınma biçimleri

bu değerlendirmede temsil edilmemektedir.

Bu nedenle V2 için doğru ifade şudur:

> Sentetik sağlık metni bozulmalarında başarılı domain adayıdır; gerçek sağlık
> ASR'si üzerinde henüz doğrulanmış üretim modeli değildir.

## 8. Başarısız veya Reddedilen Yaklaşımlardan Öğrenilenler

### 8.1 Daha Büyük veya Generative Model Tek Başına Çözüm Değildir

Temel eksik, model boyutundan çok gerçek insan-düzeltilmiş ASR çiftlerinin
azlığıdır. Büyük bir generative model daha akıcı metin üretebilir; ancak
yanlış içerik ekleme ve silme riskini de artırabilir.

### 8.2 Düşük Threshold Recall'u Artırırken Riski de Artırır

Confidence thresholdları düşürüldüğünde model daha fazla değişiklik
yapmaktadır. Ancak bu durum yanlış değişikliklerin ve doğru girdi bozulmasının
artmasına neden olur. Bu yüzden yalnızca WER'i düşüren değil, kabul edilen risk
sınırları içinde kalan threshold profilleri seçilmiştir.

### 8.3 Otomatik Hedefler İnsan Düzeltmesinin Yerini Tutmaz

Otomatik projeksiyonlar casing ve noktalama gibi içerik koruyan görevlerde
yararlıdır. Fakat yanlış tanınmış kelime, ek, özel isim ve ağır ASR hatalarının
doğru hedefini güvenilir biçimde oluşturamaz. Düşük lexical recall'un temel
nedenlerinden biri budur.

### 8.4 Açık Aksiyonlar Mantıklı Olsa da Veri Dağılımı Belirleyicidir

`KEEP`, `REMOVE` ve `SET_*` gibi açık aksiyonlar mimari olarak daha temizdir.
Ancak yalnızca aksiyon uzayını değiştirmek genel test başarısını garanti
etmemiştir. Açık aksiyonlu model bir doğru girdiyi bozduğu için reddedilmiştir.

## 9. Kullanım Önerileri

### 9.1 Genel Türkçe Kullanım

Genel ASR çıktılarında V1 General kullanılmalıdır.

Önerilen varsayılan profil:

```text
case_threshold=0.75
punctuation_threshold=0.90
lexical_threshold=0.95
```

Yanlış değişiklik maliyetinin çok yüksek olduğu uygulamalarda güvenli profil:

```text
case_threshold=0.75
punctuation_threshold=0.98
lexical_threshold=0.95
```

### 9.2 Sağlık Alanı Kullanımı

V2 Medical Synthetic yalnızca kontrollü pilot ve araştırma kullanımında
değerlendirilmelidir. Gerçek klinik karar veya hasta kaydı süreçlerinde,
gerçek sağlık ASR doğrulaması tamamlanmadan doğrudan kullanılmamalıdır.

## 10. Gelecek Çalışmalar

Modeli gerçekten kapsamlı bir Türkçe ASR post-processor haline getirmek için
en yüksek öncelikli adımlar şunlardır:

1. En az `5.000`, tercihen `20.000+` gerçek ASR çıktısını insanlar tarafından
   okunabilir hedeflere dönüştürmek.
2. Özellikle lexical, ek, Türkçe karakter, özel isim ve alan terimi hatalarını
   insan tarafından doğrulamak.
3. Gerçek sağlık seslerini veya gerçek sağlık ASR çıktı-hedef çiftlerini
   sağlayarak V2'yi yeniden eğitmek ve değerlendirmek.
4. Virgül, soru işareti, ünlem, iki nokta, noktalı virgül ve üç nokta
   sınıflarını dengeli örneklemek.
5. Metinden güvenilir biçimde kurtarılamayan örnekleri daha güçlü ASR ile
   yeniden transkribe eden audio-aware ikinci geçişe yönlendirmek.
6. Kişi, yer ve sağlık terimleri için yalnızca yüksek güvenle çalışan domain
   sözlüğü veya retrieval desteği eklemek.
7. WER/CER yanında insan okunabilirlik tercihi ve anlam koruma değerlendirmesi
   gerçekleştirmek.

## 11. Nihai Karar

Proje, genel Türkçe ASR çıktılarında güvenli okunabilirlik iyileştirmesi yapan
tek-checkpoint bir model üretmiştir. Kabul edilen **V1 General**, casing ve
sınırlı noktalama düzeltmelerinde güvenilir; lexical düzeltmede ise yüksek
precision fakat düşük recall gösteren konservatif bir modeldir.

Sağlık alanı için oluşturulan **V2 Medical Synthetic**, kontrollü sentetik
testte belirgin iyileşme sağlamıştır. Bununla birlikte gerçek sağlık ASR verisi
olmadan üretim başarısı iddia edilemez.

Sonuç olarak:

- **En başarılı ve genel kullanıma kabul edilen model:** V1 General.
- **Sağlık alanı araştırma adayı:** V2 Medical Synthetic.
- **Reddedilen genel deney:** Açık aksiyonlu V1; doğru girdi bozduğu ve kabul
  edilen V1'i geçemediği için.
- **En büyük kalan eksik:** İnsan tarafından düzeltilmiş gerçek ASR lexical
  hata çiftleri ve gerçek sağlık ASR değerlendirmesi.

## 12. Model Paketleri ve Doğrulama Değerleri

### V1 General

Model repository:

https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1

SHA-256:

```text
053418511BC825EFDC7B4DE288EBCC6CB79E70DC2107CA23C610015AB16138D0
```

### V2 Medical Synthetic

Model repository:

https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2

SHA-256:

```text
E2B2A326E46BC6C0506993077FDE3CC16D0F395B3AB284199E865E60FA3FAB03
```

## 13. İlgili Teknik Dosyalar

- `reports/release_manifest.json`
- `reports/RESULTS.md`
- `reports/MULTITASK_V3_ANALYSIS.md`
- `reports/multitask_token_editor_v3_analysis.json`
- `reports/postprocessor_v1_explicit_analysis.json`
- `reports/postprocessor_v2_medical_analysis.json`
- `audits/multitask_token_editor_v3_regressions.csv`
- `docs/PROGRESS.md`
