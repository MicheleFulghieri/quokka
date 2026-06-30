# 📋 Diagnostica Branch: test/mergecosmology vs upstream/development

## Panoramica Situazione Attuale

```
Merge-base (primo commit comune): 8cc0cb3bd
├── upstream/development: 72 commit DOPO merge-base (HEAD: 96ced0c9)
└── test/mergecosmology: 43 commit DOPO merge-base (HEAD: edd4cffb)

Statistiche:
- File modificati in upstream: 389 (7277 inseriti, 6080 cancellati)
- File divergenti tra branch: 81
```

---

## 🔴 PROBLEMI CRITICI IDENTIFICATI

### 1. **TOML Input Files Migration**
Upstream ha aggiunto **29+ file .toml** paralleli ai .in:
- `inputs/Advection.toml` (nuovo)
- `inputs/HydroWave.toml` (nuovo)
- `inputs/ParticleDeposition.toml` (nuovo)
- Eccetera...

**Tuo branch** mantiene `.in` files (formato vecchio):
- `inputs/BinaryOrbitCIC.in` (modificato, non migrato)
- `inputs/CosmoPowerSpectrum.in` (nuovo in branch, non in formato TOML)
- `inputs/ZeldovichPancake.in` (nuovo in branch, non in formato TOML)

**Conseguenza**: 
- Durante merge, non avrai conflitti diretti sui .in files
- MA il build potrebbe fallire se CMakeLists.txt attende .toml
- Runtime parsing fallirà se il parser cambia

### 2. **Modifiche Massive a File Particle (1922 righe!)**
Upstream ha cambiato:
- `src/particles/particle_accretion.hpp`
- `src/particles/particle_creation.hpp`
- `src/particles/particle_deposition.hpp`
- `src/particles/particle_update.hpp`
- `src/simulation.hpp` (504 righe cambiate!)

Il tuo branch ha ANCHE modificato questi file per supporto cosmologia.

**Conseguenza**: 
- Conflitti **garantiti** in questi file
- Potrebbero essere conflitti complessi (non lineari)

### 3. **Rimozione Ascent Support**
Upstream commit 96ced0c9 "Remove Ascent support (#1780)":
- Rimosso 2+ file ascent_actions.yaml
- Modificati CMakeLists.txt relati

Il tuo branch non ha questa rimozione → conflitto in ascent_actions.yaml

### 4. **Particles Only at 3D**
Upstream commit 1586f287: "Support particles only when compiled at 3D"

Tuo branch aggiunge cosmology ai particles → potenziale incompatibilità se particles non disabilitati in 1D/2D

---

## 📊 Comandi Diagnostica Dettagliati

### **Sezione A: Analisi Divergenza**

#### Comando 1: Trova il merge-base (primo commit comune)
```bash
git merge-base test/mergecosmology upstream/development
# Output: 8cc0cb3bd8f02714b2e45b67da66b9bc9ef96073
```
**Spiegazione**: Identifica il "punto di divisione" tra i due branch. Tutti i commit prima di questo sono comuni e identici. Dopo questo, i branch hanno histories divergenti.

#### Comando 2: Conta commit divergenti
```bash
# Commit in upstream DOPO merge-base
git log --oneline $(git merge-base test/mergecosmology upstream/development)..upstream/development | wc -l
# Output: 72

# Commit nel tuo branch DOPO merge-base
git log --oneline $(git merge-base test/mergecosmology upstream/development)..test/mergecosmology | wc -l
# Output: 43
```
**Spiegazione**: Ti mostra quanti commit divergenti hai in ogni branch. Più commit = merge più complesso.

#### Comando 3: Mostra tutti i commit in upstream
```bash
git log --oneline $(git merge-base test/mergecosmology upstream/development)..upstream/development
# Output: 72 righe di commit
```
**Spiegazione**: Lista OGNI commit che upstream ha fatto DOPO il merge-base. Leggi i messaggi per capire cosa è cambiato (breaking changes, etc).

#### Comando 4: Mostra tutti i commit nel tuo branch
```bash
git log --oneline $(git merge-base test/mergecosmology upstream/development)..test/mergecosmology
# Output: 43 righe di commit
```
**Spiegazione**: Equivalente per il tuo branch. Aiuta a capire la "storia" della cosmologia che hai aggiunto.

---

### **Sezione B: Analisi File Modificati**

#### Comando 5: File modificati solo in upstream
```bash
git diff --name-status $(git merge-base test/mergecosmology upstream/development)..upstream/development | grep "^M"
# Output: file M (modified)
```
**Spiegazione**: `M` = Modified, `A` = Added, `D` = Deleted. Mostra quali file upstream ha toccato.

#### Comando 6: File aggiunti solo in upstream
```bash
git diff --name-status $(git merge-base test/mergecosmology upstream/development)..upstream/development | grep "^A"
```
**Spiegazione**: File nuovi creati in upstream (come i .toml files).

#### Comando 7: File modificati in file CRITICO (simulation.hpp)
```bash
git diff $(git merge-base test/mergecosmology upstream/development)..upstream/development -- src/simulation.hpp | head -100
# Output: diff with +/- per ogni linea modificata
```
**Spiegazione**: Mostra ESATTAMENTE cosa è cambiato in uno specifico file (utile per anticipare conflitti).

#### Comando 8: Conta righe cambiate in file particle-related
```bash
git diff $(git merge-base test/mergecosmology upstream/development)..upstream/development -- \
  src/particles/particle_accretion.hpp \
  src/particles/particle_creation.hpp \
  src/particles/particle_deposition.hpp \
  src/particles/particle_update.hpp \
  src/simulation.hpp | wc -l
# Output: 1922 (linee di diff)
```
**Spiegazione**: Ti dice subito "questo sarà un merge casino" se vedi >1000 linee in file che ANCHE il tuo branch ha modificato.

#### Comando 9: Confronta gli stessi file tra branch
```bash
git diff test/mergecosmology upstream/development -- src/particles/particle_accretion.hpp | head -50
```
**Spiegazione**: Mostra le differenze SPECIFIC per un file tra i due branch. Ideale per trovare conflict zone esatte.

---

### **Sezione C: Analisi Merge Conflicts (Prima del Merge!)**

#### Comando 10: Dry-run merge (simula merge senza applicare)
```bash
git merge --no-commit --no-ff upstream/development
# Se fallisce, mostra i conflitti
# poi: git merge --abort   (per annullare)
```
**Spiegazione**: Finge di fare il merge ma NON lo applica. Così vedi quanti/quali conflitti avresti, senza rovinare niente.

#### Comando 11: Trova file che avranno conflitti
```bash
# Dopo il dry-run merge che fallisce:
git diff --name-only --diff-filter=U
# Output: file in conflitto
```
**Spiegazione**: `U` = Unmerged (conflitti). Mostra solo i file problematici.

#### Comando 12: Visualizza conflitti in un file specifico
```bash
git show :1:src/simulation.hpp > simulation.base.hpp      # versione merge-base
git show :2:src/simulation.hpp > simulation.ours.hpp      # versione branch locale (test/mergecosmology)
git show :3:src/simulation.hpp > simulation.theirs.hpp    # versione upstream
# Poi diff-3 per visualizzare i 3 lati del conflitto
kdiff3 simulation.base.hpp simulation.ours.hpp simulation.theirs.hpp
# oppure in terminale:
diff3 -m simulation.ours.hpp simulation.base.hpp simulation.theirs.hpp > simulation.merged.hpp
```
**Spiegazione**: Ti mostra il conflitto in 3 versioni: quella comune (base), la tua, e quella di upstream. Aiuta a capire quale cambiamento tenere.

---

### **Sezione D: Analisi Modifiche Specifiche**

#### Comando 13: Mostra cosa upstream ha fatto a File X
```bash
git show upstream/development:src/QuokkaSimulation.hpp > /tmp/upstream_quokka.hpp
git show test/mergecosmology:src/QuokkaSimulation.hpp > /tmp/local_quokka.hpp
diff -u /tmp/upstream_quokka.hpp /tmp/local_quokka.hpp | head -100
```
**Spiegazione**: Confronto line-by-line di cosa i due branch hanno fatto allo stesso file.

#### Comando 14: Mostra tutti i commit che hanno toccato File X
```bash
git log --oneline $(git merge-base test/mergecosmology upstream/development)..upstream/development -- src/simulation.hpp
git log --oneline $(git merge-base test/mergecosmology upstream/development)..test/mergecosmology -- src/simulation.hpp
```
**Spiegazione**: Mostra la "storia" di cambamenti a uno specifico file in entrambi i branch.

#### Comando 15: Mostra il contenuto di un file nel merge-base
```bash
COMMON=$(git merge-base test/mergecosmology upstream/development)
git show $COMMON:src/physics_info.hpp | head -50
# Così vedi cosa ERA prima che i due branch divergessero
```
**Spiegazione**: Il "punto di partenza" prima della divergenza. Utile per capire cosa ognuno ha aggiunte.

---

### **Sezione E: Preparazione Merge**

#### Comando 16: Crea backup branch prima di merge
```bash
git branch test/mergecosmology-backup
# Ora hai una copia di sicurezza intera del branch
```
**Spiegazione**: Sempre fare backup prima di operazioni rischiose come merge.

#### Comando 17: Aggiorna il reference locale
```bash
git fetch upstream development
# Assicura che hai la versione LATEST di upstream/development
```
**Spiegazione**: `git fetch` aggiorna i remote-tracking branches senza modificare il tuo codice locale.

#### Comando 18: Merge strategy - 3-way merge verbose
```bash
git merge -X ours upstream/development --verbose
# -X ours = in caso di conflitti automatici, preferisci "ours" (test/mergecosmology)
# --verbose = stampa info extra
```
**Spiegazione**: `-X ours` è una strategia che risolve automaticamente alcuni conflitti in tuo favore. Utile se sei fiducioso che i tuoi cambiamenti sono corretti.

---

### **Sezione F: Post-Merge Validation**

#### Comando 19: Verifica numero di conflitti residui
```bash
git diff --name-only --diff-filter=U | wc -l
# Output: numero di file ancora in conflitto
```
**Spiegazione**: Dopo aver fatto merge e risolto alcuni conflitti, ti dice quanti ne restano da risolvere.

#### Comando 20: Compila dopo merge
```bash
rm -rf build && mkdir build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3 -G Ninja
ninja -j6 2>&1 | tee build.log
# Se compila senza errori, il merge è OK
```
**Spiegazione**: Il test finale: **se compila, il merge è buono** (modulo bugs di runtime).

#### Comando 21: Testa problemi cosmologia specificamente
```bash
ctest -R "Cosmo|Zeldovich" --output-on-failure
# Output: PASS o FAIL per ogni test cosmologia
```
**Spiegazione**: Verifica che i tuoi problemi cosmologia NON sono stati rotti dal merge.

#### Comando 22: Testa sample di non-cosmologia problems
```bash
ctest -R "Advection|HydroBlast" --output-on-failure
# Assicura che non hai rotto niente di upstream
```
**Spiegazione**: Regression check per garantire che il merge non ha rotto altre cose.

---

## 📝 Script Completo: Diagnostica Automatica

```bash
#!/bin/bash
# script: diagnose_merge.sh

BRANCH_LOCAL="test/mergecosmology"
BRANCH_UPSTREAM="upstream/development"

echo "=== BRANCH DIAGNOSTICA ==="
echo "Local branch: $BRANCH_LOCAL"
echo "Upstream branch: $BRANCH_UPSTREAM"
echo ""

MERGE_BASE=$(git merge-base $BRANCH_LOCAL $BRANCH_UPSTREAM)
echo "Merge-base (common ancestor): $MERGE_BASE"
echo ""

echo "=== COMMIT DIVERGENCE ==="
UPSTREAM_COMMITS=$(git log --oneline $MERGE_BASE..$BRANCH_UPSTREAM | wc -l)
LOCAL_COMMITS=$(git log --oneline $MERGE_BASE..$BRANCH_LOCAL | wc -l)
echo "Commits in upstream since fork: $UPSTREAM_COMMITS"
echo "Commits in local branch: $LOCAL_COMMITS"
echo ""

echo "=== FILE STATISTICS ==="
echo "Files modified upstream:"
git diff --name-status $MERGE_BASE..$BRANCH_UPSTREAM | wc -l
echo "Files modified locally:"
git diff --name-status $MERGE_BASE..$BRANCH_LOCAL | wc -l
echo "Files that diverge (both modified):"
comm -12 \
  <(git diff --name-only $MERGE_BASE..$BRANCH_UPSTREAM | sort) \
  <(git diff --name-only $MERGE_BASE..$BRANCH_LOCAL | sort) | wc -l
echo ""

echo "=== CRITICAL FILE ANALYSIS ==="
CRITICAL_FILES=(
  "src/simulation.hpp"
  "src/QuokkaSimulation.hpp"
  "src/physics_info.hpp"
  "src/particles/particle_update.hpp"
  "src/particles/particle_deposition.hpp"
)

for file in "${CRITICAL_FILES[@]}"; do
  UPSTREAM_CHANGES=$(git diff $MERGE_BASE..$BRANCH_UPSTREAM -- "$file" 2>/dev/null | wc -l)
  LOCAL_CHANGES=$(git diff $MERGE_BASE..$BRANCH_LOCAL -- "$file" 2>/dev/null | wc -l)
  if [[ $UPSTREAM_CHANGES -gt 0 ]] && [[ $LOCAL_CHANGES -gt 0 ]]; then
    echo "⚠️  CONFLICT RISK: $file"
    echo "   - Upstream changes: $UPSTREAM_CHANGES lines"
    echo "   - Local changes: $LOCAL_CHANGES lines"
  fi
done
echo ""

echo "=== RECENT UPSTREAM COMMITS ==="
git log --oneline $MERGE_BASE..$BRANCH_UPSTREAM | head -10
echo ""

echo "=== RECENT LOCAL COMMITS ==="
git log --oneline $MERGE_BASE..$BRANCH_LOCAL | head -10

```

**Uso**:
```bash
chmod +x diagnose_merge.sh
./diagnose_merge.sh
```

---

## 🎯 Checklist Merge Procedure

### BEFORE MERGE
- [ ] `git branch test/mergecosmology-backup` (backup)
- [ ] `git fetch upstream development` (aggiorna)
- [ ] `./diagnose_merge.sh` (diagnostica)
- [ ] Leggi i 10 commit più recenti di upstream
- [ ] Verifica CMakeLists.txt per TOML changes
- [ ] Backup della cartella build/

### DURING MERGE
```bash
git merge --no-commit --no-ff upstream/development
# Risolvi manualmente ogni conflitto in file critici
# Usa 3-way merge tool (kdiff3, meld, etc.)
git add <file-risolto>
# Ripeti per ogni file
git commit -m "Merge upstream/development with cosmology"
```

### AFTER MERGE
- [ ] `cmake` build da clean
- [ ] `ctest -R "Cosmo*"` (testa cosmologia)
- [ ] `ctest -R "Advection*"` (testa non-cosmologia sample)
- [ ] Verifica che ZeldovichPancake ha CMakeLists.txt
- [ ] Verifica input files (TOML vs .in)
- [ ] `git log --oneline -5` (verifica history)

---

## ⚠️ Problemi Known

1. **HydroWave.in rimosso nel branch**: Upstream ha `.toml`, branch ha `.in`. Decision: quale mantenere?

2. **ZeldovichPancake CMakeLists mancante**: Aggiungere a src/problems/ZeldovichPancake/

3. **Cosmology input files non in TOML**: Se build richiede .toml, bisogna convertire CosmoPowerSpectrum.in, etc.

4. **Ascent support rimosso upstream**: Verifica che branch non usa Ascent

5. **Physics_Traits vs PhysicsTraits dualità**: Decide quale usare per nuovo codice

---

## 📖 Risorse Utili

- Git merge docs: `git help merge`
- 3-way merge visualization: `kdiff3`, `meld`, `P4Merge`
- Conflict resolution: `git config merge.tool <tool>`
- Rebase alternative: `git rebase upstream/development` (history lineare)

