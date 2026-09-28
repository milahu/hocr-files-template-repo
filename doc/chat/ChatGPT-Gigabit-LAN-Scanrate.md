# Gigabit LAN Scanrate

**User:** Anonymous  
**Created:** 2026/9/28 8:34:22  
**Updated:** 2026/9/28 8:38:07  
**Exported:** 2026/9/28 8:40:22  
**Link:** [<https://chatgpt.com/c/6aba0a6e-5210-83eb-990e-0bd0490ffbeb>](https://chatgpt.com/c/6aba0a6e-5210-83eb-990e-0bd0490ffbeb)

## Prompt:

9/28/2026, 8:34:56 AM

ab welcher scan-geschwindigkeit (seiten pro minute) braucht ein 600dpi dokumentenscanner gigabit-LAN?

am meisten traffic braucht: 600dpi, farbe, duplex, DIN A3

## Response:

9/28/2026, 8:34:58 AM

Wenn wir vom **Worst Case ohne Kompression** ausgehen – **DIN A3, 600 dpi, Farbe (24 Bit), Duplex** – lässt sich die benötigte LAN-Bandbreite ziemlich gut berechnen.

### Rohdatenmenge

DIN A3 = **11,69 × 16,54 Zoll**

Bei 600 dpi:

- Auflösung: ca. **7.014 × 9.921 Pixel**
- Farbe: **24 Bit = 3 Byte/Pixel**
- pro Seite: ca. **208,8 MB** unkomprimierte Bilddaten
- Duplex: ca. **417,6 MB pro A3-Blatt**

Für Gigabit-Ethernet stehen theoretisch **1.000 Mbit/s = 125 MB/s** zur Verfügung. Realistisch sollte man wegen Ethernet/IP/TCP-Overhead und Reserven eher mit etwa **110–120 MB/s** Nutzdaten rechnen.

### Daraus ergibt sich

| Scan-Geschwindigkeit | Datenrate bei A3, 600 dpi, Farbe, Duplex | Gigabit-LAN |
|---:|---:|----|
| 10 Seiten/min | ~69,6 MB/s | problemlos |
| 15 Seiten/min | ~104,4 MB/s | wird knapp |
| **16 Seiten/min** | **~111,4 MB/s** | praktisch Grenze |
| 20 Seiten/min | ~139,2 MB/s | **\> 1 GbE** |
| 30 Seiten/min | ~208,8 MB/s | **\> 1 GbE** |
| 40 Seiten/min | ~278,4 MB/s | **\> 1 GbE** |
| 60 Seiten/min | ~417,6 MB/s | **\> 1 GbE** |
| 100 Seiten/min | ~696 MB/s | **\> 1 GbE** |

**Wichtig:** Ich interpretiere „Seiten/min“ hier als **A3-Blätter/min**, also bei Duplex zwei gescannte Seiten pro Blatt.

### Kurzantwort

Bei diesen Extremparametern erreicht man die theoretische Grenze von **1-Gbit-LAN bei etwa 17,9 A3-Blättern/min**.

Für einen Scanner, der z. B. **20 A3-Blätter/min Duplex bei 600 dpi Farbe** tatsächlich als unkomprimierte Daten übertragen müsste, wäre **Gigabit-LAN bereits zu langsam**. Dafür bräuchte man mindestens 2,5 GbE, wobei 2,5 GbE bis ungefähr **44 Blatt/min** reicht.

**In der Praxis ist die Situation aber deutlich günstiger**, weil Dokumentenscanner die Bilddaten typischerweise komprimieren (z. B. JPEG/CCITT/JBIG bzw. herstellerspezifische Verfahren). Deshalb ist die maximale physikalische Scangeschwindigkeit **nicht automatisch** die Geschwindigkeit, ab der Gigabit-LAN zum Flaschenhals wird.

Wenn du mir **das konkrete Scanner-Modell** nennst, kann ich dir aus dessen Datenblatt berechnen, ob **1 GbE, 2,5 GbE oder 10 GbE** für den maximalen 600-dpi-A3-Duplex-Betrieb sinnvoll ist.

## Prompt:

9/28/2026, 8:37:46 AM

du hast mich falsch verstanden.  
mit meiner frage "ab welcher scan-geschwindigkeit (seiten pro minute) braucht ein 600dpi dokumentenscanner gigabit-LAN?" habe ich gemeint:  
welche maximale scan-geschwindigkeit kann so ein scanner erreichen, wenn er die daten nur mit 100Mbit/s übertragen kann?

bitte vergleiche:  
100Mbit/s vs 1Gbit/s vs 10Gbit/s  
DIN A3 vs DIN A4  
farb-scan vs graustufen-scan vs schwarzweiss-scan

## Response:

9/28/2026, 8:37:49 AM

> **Calculating theoretical scan page rates**
>
> ------------------------------------------------------------------------
>
> **Calculating theoretical scan page rates**
>
> Worked for 7s

Ja - jetzt ist die Frage klar. Du möchtest wissen, **wie viele 600-dpi-Seiten pro Minute maximal über die jeweilige Netzwerkverbindung übertragen werden können**, wenn wir zunächst von **unkomprimierten Rohdaten** ausgehen.

Dabei ist wichtig: Bei **Duplex** verdoppelt sich der Datenstrom, weil pro Blatt zwei Seiten übertragen werden.

### Annahmen

- 600 dpi
- A4: 210 × 297 mm
- A3: 297 × 420 mm
- Farbe: 24 Bit/Pixel
- Graustufen: 8 Bit/Pixel
- Schwarzweiß: 1 Bit/Pixel
- **keine Kompression**
- 100 Mbit/s, 1 Gbit/s und 10 Gbit/s werden als nutzbare Datenrate betrachtet
- Werte sind theoretische Maximalwerte; Protokoll-Overhead und Scanner-interne Datenformate können die tatsächlichen Werte etwas reduzieren.

## Maximale Seiten/min bei 600 dpi

| Format | Scanmodus          |      100 Mbit/s |        1 Gbit/s |        10 Gbit/s |
|--------|--------------------|----------------:|----------------:|-----------------:|
| **A4** | Farbe, 24 Bit      |   **7,2 S/min** |  **71,8 S/min** |    **718 S/min** |
| **A4** | Graustufen, 8 Bit  |  **21,6 S/min** | **215,5 S/min** |  **2.155 S/min** |
| **A4** | Schwarzweiß, 1 Bit | **172,4 S/min** | **1.724 S/min** | **17.240 S/min** |
| **A3** | Farbe, 24 Bit      |   **3,6 S/min** |  **35,9 S/min** |    **359 S/min** |
| **A3** | Graustufen, 8 Bit  |  **10,8 S/min** | **107,8 S/min** |  **1.077 S/min** |
| **A3** | Schwarzweiß, 1 Bit |  **86,2 S/min** |   **862 S/min** |  **8.620 S/min** |

### Besonders relevant für Dokumentenscanner

Bei **A3, 600 dpi, Farbe** ist das Netzwerk tatsächlich sehr schnell ein limitierender Faktor:

- Eine A3-Farbseite enthält bei 600 dpi etwa **69,6 Millionen Pixel**.
- Bei 24 Bit sind das etwa **208,8 MB pro Seite**.
- Bei **100 Mbit/s** können davon theoretisch nur **3,6 Seiten/min** übertragen werden.
- Bei **1 Gbit/s** sind es etwa **35,9 Seiten/min**.
- Bei **10 Gbit/s** etwa **359 Seiten/min**.

Für **Duplex** halbiert sich die mögliche Blattgeschwindigkeit:

| A3, 600 dpi, Farbe      |     100 Mbit/s |        1 Gbit/s |      10 Gbit/s |
|-------------------------|---------------:|----------------:|---------------:|
| **Simplex**             | 3,6 Seiten/min | 35,9 Seiten/min | 359 Seiten/min |
| **Duplex: Blätter/min** |        **1,8** |        **18,0** |      **179,6** |

Das heißt beispielsweise:

> Ein Scanner mit **100 Mbit/s Ethernet** kann bei A3/600 dpi/Farbe theoretisch nur etwa **1,8 A3-Blätter/min im Duplexbetrieb** übertragen, wenn die Daten unkomprimiert sind.

Bei **1 Gbit/s** sind es dagegen bereits etwa **18 A3-Blätter/min Duplex**.

------------------------------------------------------------------------

## Aber: Schwarzweiß ist ein völlig anderer Fall

Bei echtem **1-Bit-Schwarzweiß** ist die Datenmenge gegenüber Farbe um den Faktor **24 kleiner**.

Deshalb könnte selbst 100 Mbit/s theoretisch enorme Scanraten übertragen:

**A3, 600 dpi, 1 Bit:**

- 100 Mbit/s → **86 Seiten/min**
- 1 Gbit/s → **862 Seiten/min**
- 10 Gbit/s → **8.620 Seiten/min**

Das Netzwerk ist bei einem normalen Dokumentenscanner also **nicht automatisch bei jeder Scanart der Flaschenhals**.

### Und genau hier kommt Kompression ins Spiel

Die obige Tabelle ist der **Worst Case für unkomprimierte Bilddaten**. Ein realer Dokumentenscanner wird die Daten normalerweise komprimieren. Bei Schwarzweiß können beispielsweise sehr effiziente Verfahren verwendet werden, während bei Farbe typischerweise JPEG-artige Kompression zum Einsatz kommt.

Dadurch kann ein Scanner beispielsweise **40, 60 oder 100 Seiten/min** schaffen, obwohl die theoretischen Rohdaten weit oberhalb von 1 Gbit/s liegen.

Wenn du also konkret die Frage beantworten willst **„Reicht 100 Mbit/s für einen modernen Dokumentenscanner mit 600 dpi und z. B. 60/80/100/120 ppm?“**, müsste man als nächsten Schritt die **typischen komprimierten Datenraten** betrachten. Das ist für die Auswahl von **100M/1G/10G Ethernet** wesentlich realistischer als die Rohdatenrechnung oben.

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
