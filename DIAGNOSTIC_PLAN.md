# Piano Diagnostico: Deformazione Sfera CosmoSphereDM

## 1. TEST DI ISOLAMENTO CAUSA
### 1.1 Velocità nulla (GIÀ FATTO ✓)
- **Cosa**: vx = 0 → sparisce deformazione
- **Conclusione**: Non è instabilità statica, è legato al moto

### 1.2 Profilo di densità: Sharp vs Smooth
**Test A**: Profilo sharp (attuale)
```cpp
rho = (r <= R_sphere) ? rho_max : rho_min;
```

**Test B**: Profilo smooth ORIGINALE (più corto)
```cpp
R_smooth = 1.0e22;  // 1/3 dell'originale
rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
```

**Misura**: Traccia la deformazione vs R_smooth
- Se cresce linearmente → problema è pendenza densità
- Se satura → problema è oscillazioni PPM pure

---

## 2. TEST DI RISOLUZIONE E CONVERGENZA
### 2.1 Convergenza della deformazione
Crea input variati:
```
256x256x256 (attuale)
128x128x128 (metà)
512x512x512 (doppio)
1024x1024x1024 (quadruplo)
```

**Misura**: 
- Massimo shift della densità lungo x vs risoluzione
- Se deformazione ∝ 1/N → è errore PPM discretizzazione
- Se converge → è problema fisico
- Se cresce → è instabilità che si accentua

---

## 3. TEST DEL METODO NUMERICO
### 3.1 Variazione CFL
```
quokka.cfl = 0.05, 0.1, 0.2, 0.3, 0.5
```

**Misura**:
- CFL basso riduce errore PPM? Quanto?
- Se CFL=0.05 → sparisce → è errore temporale
- Se rimane → non è semplice stabilità

### 3.2 Cercare il limitatore di slope
**Dove**: `src/hydro/HyperbolicSystem.hpp` o `src/hydro/slope_limiters.hpp`

**Cosa cercare**:
```cpp
// Cerca: VanLeer, MinMod, MonotonizedCentral, Superbee
// Cambia e ricompila
```

**Test con limitatori diversi**:
- Se MinMod riduce → è oscillazione PPM
- Se rimane uguale → è problema diverso

---

## 4. TEST DI OUTPUT DIAGNOSTICO
### 4.1 Salva state intermedi
**Cosa aggiungere al codice**:
```cpp
// Nella setInitialConditions, stampa:
amrex::Print() << "Initial state center:\n";
for (int i = -5; i <= 5; i++) {
    int idx = n_cells_x/2 + i;
    amrex::Print() << "  ρ[" << idx << "] = " << rho_profile[idx] << "\n";
}
```

**Misura**: Asimmetria della densità iniziale lungo x
- Se C6-C4 ≠ C4-C2 inizialmente → problema setup
- Se sono uguali → problema evoluzione

### 4.2 Traccia energia interna nel tempo
```cpp
// Aggiungi a problem_main()
// Ad ogni plotfile: E_int_max, E_int_avg, E_int_centro
```

**Misura**: Dove cresce e_int?
- Se solo nel gradiente ρ → è PPM
- Se uniforme → è coupling diverso

### 4.3 Salva velocità e divergenza
```cpp
// Traccia div(v) nella sfera
// Se div(v) ≠ 0 al centro → problema accoppiamento
```

---

## 5. TEST DI PHYSICS
### 5.1 Pressione iniziale minima
Commenta/decomenta:
```cpp
// Versione A: NO pressione
// E_int = 0

// Versione B: Pressione minima di background
amrex::Real P = 1.0e-16;
E_int = P / ((gamma - 1.0) * rho);

// Versione C: Pressione in equilibrio idrostatico (teorico)
// P(r) = P_centro * (ρ(r)/ρ_centro)^(4/3)  [isotermo]
```

**Misura**: Quale caso deforma meno?
- Se B << A → pressione stabilizza → è artefatto
- Se A ≈ B ≈ C → è instabilità fisica

### 5.2 Confronto con ambiente omogeneo
**Test**: Metti sfera in mezzo con stessa pressione/temperatura
```cpp
rho_out = rho_max;  // non rho_min
P_out = P_sfera;
vx_out = drift_vel;
```

**Misura**: Se la sfera resta stabile → è instabilità KH vera
- Se deforma lo stesso → è artefatto numerico

---

## 6. ANALISI POST-PROCESSAMENTO
### 6.1 Centroide della densità
```python
# Script Python per analizzare plotfiles
import yt
ds = yt.load("plotfile_XXXXX")
data = ds.all_data()
rho = data['density']
x = data['x']
y = data['y']
z = data['z']

# Centro di massa
x_cm = (rho * x).sum() / rho.sum()
y_cm = (rho * y).sum() / rho.sum()
z_cm = (rho * z).sum() / rho.sum()

print(f"Shift lungo x: {x_cm - x_centro_teorico}")
print(f"Shift lungo y: {y_cm - y_centro_teorico}")
print(f"Shift lungo z: {z_cm - z_centro_teorico}")

# Profilo di densità
r = np.sqrt((x-x_cm)**2 + (y-y_cm)**2 + (z-z_cm)**2)
rho_profile = rho[np.argsort(r)]
```

### 6.2 Power spectrum delle perturbazioni
```python
# FFT della densità attorno alla sfera
# Se crescono modi con k_x >> k_y → è artefatto numerico direzionale
```

### 6.3 Asimmetria della densità
```python
# Prendi slice a y=z=centro
# Plot ρ(x) vs x
# Misura: max_left vs max_right
asymmetry = (rho_right - rho_left) / rho_medio
print(f"Asimmetria: {asymmetry}")
```

---

## 7. ORDINE DI ESECUZIONE CONSIGLIATO

1. **Fase 1** (veloce, <1 ora):
   - 2.1 Convergenza (3-4 risoluzioni)
   - 3.1 CFL (5 valori)
   - → Capisce se è temporale o spaziale

2. **Fase 2** (media, 2-3 ore):
   - 1.2 Sharp vs Smooth a risoluzione fissa
   - 4.1 Output diagnostico iniziale
   - → Isola se è setup o evoluzione

3. **Fase 3** (diagnostica profonda):
   - 5.1 Test pressione
   - 5.2 Ambiente omogeneo
   - 6.1-6.3 Analisi post-processing

4. **Fase 4** (fix):
   - Cambia limitatore PPM
   - Ottimizza smooth radius
   - Aggiungi pressione di base se necessario

---

## 8. COSA MISURARE ESATTAMENTE

Per ogni test, salva in un file:
```
risoluzione, CFL, R_smooth, t_simulazione
max_deformazione_x, max_deformazione_y, max_deformazione_z
E_int_max, E_int_avg
asymmetry_factor
```

Poi plot:
```python
# deformazione_x vs risoluzione
# deformazione_x vs CFL
# deformazione_x vs R_smooth
```

Il pattern del plot dirà quale parametro controlla il problema.
