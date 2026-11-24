# strata-m4: Mistral Small 4 auf einer V100 und zwei Xeon-Sockeln

strata-m4 ist ein eigener Modellpfad im Strata-Fork, geschrieben für Mistral Small 4 (119B gesamt, 6,5B aktiv, 128 Experten mit Top-4 plus einem Shared-Experten, MLA-Attention, Vision). Die quantisierten Rechenkerne (GGUF-Blockformate wie in llama.cpp, gruppierte Experten auf der GPU, AVX-512-Expertenzeilen auf der CPU) kommen aus Strata. Neu geschrieben sind:

- MLA mit int8-Latent-Cache,
- Router und Kopplung zwischen GPU und Host,
- NUMA-Pool,
- Expertenplatzierung,
- Spekulation,
- Server.

## Speicherhierarchie

| Ebene | Inhalt | Wer rechnet |
| --- | --- | --- |
| VRAM (V100, 32 GB) | alle dichten Gewichte (MLA-Projektionen, Shared Expert, Router, Kopf, Embeddings), int8-MLA-KV-Cache (11,5 KiB pro Token), die heißesten Experten als Slots, Scratch | GPU |
| Node der GPU (96 GB) | die nächstheißen Experten (`mbind` auf den Node, Transparent Huge Pages) | Kerne dieses Nodes |
| anderer Node (96 GB) | die kältesten Experten und die globale n-Gramm-Tabelle | Kerne **dieses** Nodes (Near-Data: über UPI laufen nur Aktivierungen und Ergebnisse, keine Gewichte) |
| Datei (NVMe/Page Cache) | Experten, die nirgends mehr hineinpassen, direkt aus der gemappten GGUF | anderer Node |

Die Verteilung richtet sich nach gemessener Hitze. Jede Route jeder Schicht wird gezählt und beim Beenden nach `<modell>.m4heat` geschrieben. Beim nächsten Start liegen die meistgenutzten Experten im VRAM, die nächsten auf dem GPU-Node und der Rest auf dem anderen Node. Der erste Start verteilt deshalb noch gleichmäßig, ab dem zweiten wird es besser.

Mit `--numa split` gibt es die Alternative: Jeder CPU-Experte wird zeilenweise halbiert, und jede Hälfte liegt auf einem Node und wird dort gerechnet. Damit arbeiten beide Speichercontroller an jedem Experten. Dafür kostet jede Phase eine Barriere über beide Sockel. Messen Sie beide Varianten (`bench.sh`).

Bei Q4_K_M (ca. 67 GB) passt in der Regel alles, was nicht im VRAM liegt, in die 96 GB des GPU-Nodes. Der zweite Node hält dann nur die n-Gramm-Tabelle, und es läuft praktisch kein Gewichtsverkehr über UPI.

## Ablauf eines Tokens

Ein Decode-Fenster (aktuelles Token plus bis zu 15 n-Gramm-Entwürfe) ist ein einziger CUDA-Graph. Pro Schicht läuft Folgendes ab:

1. Die GPU rechnet die MLA-Attention: q über `wk_b` absorbiert, Scores direkt gegen den int8-Latent-Cache, Ausgabe über `wv_b`.
2. Der Router bestimmt Top-4 per Softmax, mit Normierung und Skala aus der GGUF.
3. Die GPU veröffentlicht die Routen und die normierte Eingabe in gemapptem Host-Speicher.
4. **Gleichzeitig** rechnet die GPU den Shared Experten und ihre VRAM-Treffer, und der NUMA-Pool rechnet die Fehltreffer auf dem jeweils zuständigen Node.
5. Ein Warte-Kernel hält die GPU an, bis der Host fertig meldet. Danach werden die Ergebnisse kombiniert.

Prompts laufen in Chunks (`--chunk`, Standard 512). Die Attention läuft dort über cuBLAS-Tensorkerne (f16), und die Fehltreffer gehen gesammelt an die CPU.

## Installation auf dem Server

```sh
tar xzf strata-m4.tar.gz && cd strata
./mistral4/install.sh --model /pfad/Mistral-Small-4-119B-2603-Q4_K_M.gguf \
    [--mmproj /pfad/mmproj-F16.gguf] [--tokenizer /pfad/tekken.json] [--systemd]
```

Das Skript prüft Treiber und GPU. Es sucht ein CUDA-12.x-Toolkit, denn CUDA 13 kann kein Volta mehr. Wenn keines da ist, bietet es nach Rückfrage `cuda-toolkit-12-9` aus NVIDIAs Repo an; der Treiber 580 bleibt dabei unverändert. Danach baut es:

- `strata-m4` für sm_70,
- llama.cpp in einem festen Commit, als Referenz und für libmtmd,
- `m4-vision`.

Außerdem legt es ein venv mit `mistral-common` an und schreibt `~/strata-m4/m4.env`.

## Prüfen und messen

```sh
~/strata-m4/parity.sh       # Perplexity: llama.cpp gegen strata-m4 (--kv f16 und int8) auf denselben Token-IDs
~/strata-m4/bench.sh        # Prompt- und Decode-Tempo: llama.cpp (--n-cpu-moe) gegen strata-m4
```

`parity.sh` kommt zuerst. Mit f16-Cache sollten die Werte innerhalb von etwa 0,5 % liegen, mit int8 innerhalb von etwa 1,5 %. Liegt die Abweichung bei mehreren Prozent, ist das ein Fehler: Bitte das Log schicken.

## Betrieb

```sh
~/strata-m4/venv/bin/python ~/strata-m4/m4serve.py --engine ~/strata-m4/bin/strata-m4 \
    --model "$M4_MODEL" --tokenizer "$M4_TOKENIZER" --engine-args "--ctx 65536 --spec 6" --port 8080
# oder: sudo systemctl enable --now strata-m4
```

Die API ist OpenAI-kompatibel und bietet `/v1/chat/completions` (mit Streaming), `/v1/completions`, `/v1/models` und `/health`. Unterstützt werden:

- `reasoning_effort` (none / high), über die Antwortfelder `reasoning_content` und `content`,
- `tools` und `tool_calls`,
- Bilder (`image_url` als Daten-URL, Datei oder http).

Den Prompt baut `mistral-common` aus der Anfrage, also Mistrals eigene Referenz für das Chatformat.

Wichtige Engine-Optionen (`strata-m4 --help`):

| Option | Bedeutung |
| --- | --- |
| `--ctx N` | Kontext; int8-Cache 11,5 KiB/Token (256K = 2,9 GB) |
| `--kv int8\|f16` | Format des Latent-Caches |
| `--spec N` | n-Gramm-Entwürfe pro Fenster (0–15) |
| `--numa tiered\|split\|off` | Platzierung der CPU-Experten |
| `--ram-gpu-node`, `--ram-other-node` | Arena-Größen in GiB |
| `--vram-reserve`, `--vram-experts` | VRAM-Reserve und Obergrenze für Expertenslots |
| `--workers N`, `--host-core N` | Worker pro Node; der Kern des GPU-fütternden Host-Threads |
| `--ngram-mib N` | globale n-Gramm-Tabelle auf dem anderen Node |
| `--attn tc\|kernel\|gemm` | Decode-Attention: Tensor-Kerne (Standard), CUDA-Kerne, cuBLAS |
| `--attn-temp-scale X` | Attention-Temperatur ab 8192 Tokens erzwingen (Mistral: 0,1; die unsloth-GGUF hat keine) |
| `--eagle PFAD` | Eagle-Kopf (Ordner mit `consolidated.safetensors` und `params.json`) |
| `--eagle-k N`, `--eagle-p P` | Eagle: höchstens N Entwürfe pro Fenster; Abbruch, wenn die Kettenwahrscheinlichkeit unter P fällt |

## Prompt-Streaming (wie Strata)

Ab `--pf-min` Tokens (Standard 768) rechnet ein Prompt-Abschnitt alle Experten auf der GPU: die im VRAM aus ihrem Slot,
die übrigen werden aus den beim Treiber gepinnten RAM-Bereichen beider NUMA-Nodes per DMA in einen Ring von VRAM-Slots
kopiert (Kopier-Stream, die Hälften der split-Aufteilung werden dabei zusammengesetzt), während der Rechen-Stream jedes
Experten nach f16 entpackt und Gate/Up, SwiGLU und Down als Tensor-Core-GEMMs rechnet.  PCIe Gen3 x16 schafft die
~37 GB CPU-Experten in ~3 s: Abschnitte von 2048 Tokens (Standard mit Streaming) lesen damit grob 700 tok/s, 4096 grob
1300 tok/s.  Kleinere Abschnitte (Agenten-Schritte) rechnet weiter der CPU-Pool.  Dichte Produkte ab 64 Zeilen laufen
ebenfalls als f16-GEMM (`--pf-gemm`).  Schalter: `--pf-stream 0|1`, `--pf-min N`, `--pf-ring N`, `--chunk N`.

## Der Befehl `strata-m4`

`install.sh` legt `~/.local/bin/strata-m4` an (wie Stratas `strata`):

| Befehl | Was er tut |
|---|---|
| `strata-m4 launch` | Server im Hintergrund starten (falls er nicht läuft), auf Bereitschaft warten, Monitor öffnen; `--no-monitor` nur starten |
| `strata-m4 serve` | Server im Vordergrund (Strg+C beendet) |
| `strata-m4 stop` / `restart` | sauber beenden (SIGINT, nach 30 s erzwungen) / neu starten |
| `strata-m4 status [--json]` | Zustand, Modell, Kontext, Warteschlange, Tempo, GPU, Platzierung |
| `strata-m4 monitor [--once]` | Stratas Monitor mit Port und Key aus `m4.env` |
| `strata-m4 logs [-f]` | Server-Log (`~/strata-m4/logs/server.log`) |
| `strata-m4 config show \| edit \| set NAME WERT \| args add\|remove OPTION [WERT]` | `m4.env` ansehen/ändern (Key wird verdeckt) |
| `strata-m4 doctor` | Prüfungen: GPU, Dateien, venv, Monitor, Port, PATH |
| `strata-m4 bench \| speed \| sanity \| parity \| longctx` | Messskripte (nur bei gestopptem Server) |

Beispiele: `strata-m4 config args add --chunk 4096`, `strata-m4 config set M4_PORT 8095`, danach `strata-m4 restart`.

## Strata Monitor

m4serve liefert Strata's Telemetrie-Schnittstelle (`GET /v1/telemetry`, Schema 1, und `/v1/status`). Der Monitor aus dem
Strata-CLI zeigt die Engine damit wie einen Strata-Server: Slot, Warteschlange, Antwort- und Prompt-Tempo, letzte
Anfragen, GPU (nvidia-smi) und RAM.

    strata monitor --url http://127.0.0.1:8080      # oder STRATA_URL setzen; auf Port 8095 geht es ohne --url

Mit API-Key: `strata monitor --url … --credential-env M4_API_KEY`.

## Stand und offene Punkte

- **Ungetestet auf Hardware.** Alles ist für sm_70 kompiliert und geprüft (ptxas: keine Spills, 45,8 KB Shared Memory im Decode-Kernel). Der NUMA-Pool ist gegen eine Float-Referenz getestet, beide Policies. Gelaufen ist es aber noch nicht auf einer V100. Die erste Runde auf dem Server ist `parity.sh`.
- **Eagle-Drafter** (`Mistral-Small-4-119B-2603-eagle`, `--eagle PFAD`): eingebaut, noch in der Erprobung. Der Kopf (2 MLA-Schichten, dichte FFN, float8 → Q8_0, ca. 0,5 GB VRAM) läuft hinter jedem Fenster über die angenommenen Zeilen und entwirft eine Kette von bis zu `--eagle-k` Tokens (Standard 3), solange das Produkt der Wahrscheinlichkeiten über `--eagle-p` (Standard 0,3) liegt. Findet der n-Gramm-Lookup im Verlauf einen Treffer (Code-Bearbeitung), hat dieser Vorrang. Prüfen: `strata-m4 ... --eagle PFAD --eagle-acc ids.txt` (Treffsicherheit des Kopfes bei vorgegebenem Text).
- **Adaptives Umschichten** der Experten im laufenden Betrieb fehlt noch. Die Platzierung folgt der Hitze-Datei des letzten Laufs.
- **Lange Prompts:** Fehltreffer gehen an die CPU. Ab einigen tausend Token pro Chunk wäre es schneller, die Experten über PCIe in VRAM-Staging-Puffer zu streamen. Das ist geplant.
- Bilder laufen über llama.cpps mtmd (Pixtral-Tower aus der mmproj-GGUF). Mit `--vision-cpu` belegen sie kein VRAM.
