

import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import Counter, OrderedDict
from itertools import groupby
import zlib
from datetime import datetime
import re
from scipy import stats

# Versuche tkinter zu importieren (optional für GUI)
HAS_TKINTER = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
    HAS_TKINTER = True
except ImportError:
    pass

# Versuche matplotlib zu importieren
HAS_MATPLOTLIB = False
try:
    import matplotlib
    if HAS_TKINTER:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    if HAS_TKINTER:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
except ImportError:
    pass


# ============================================================================
# FARBEN (Trauth Research LLC)
# ============================================================================
COLORS = {
    'gold': '#D4A574',
    'gold_dark': '#B8935F',
    'gold_light': '#E6C8A0',
    'dark': '#1A1A1A',
    'gray': '#666666',
    'light_gray': '#F8F8F8',
    'white': '#FFFFFF',
    'bg_plot': '#FDF6EC',
    'plot_color': '#4374B3'
}


# ============================================================================
# BITSTRING-METHODEN
# ============================================================================
BITSTRING_METHODS = {
    'v0': {
        'name': 'Summe→Vorzeichen',
        'short': 'Summe',
        'description': 'Original-Methode: Summe über alle Iterationen, positiv=1, negativ=0'
    },
    'v1': {
        'name': 'Wendepunkt-Dominanz',
        'short': 'Wendepunkt',
        'description': 'Mehr Aufwärts- als Abwärts-Wendepunkte = 1'
    },
    'v2': {
        'name': 'Skewness (Schiefe)',
        'short': 'Skewness',
        'description': 'Rechtsschiefe (>0) = 1, Linksschiefe = 0'
    },
    'v3': {
        'name': 'Median→Vorzeichen',
        'short': 'Median',
        'description': 'Median > 0 = 1, sonst 0'
    },
    'v4': {
        'name': 'FFT-Dominanz',
        'short': 'FFT',
        'description': 'Dominante Frequenz über Median aller Neuronen = 1'
    },
    'v5': {
        'name': 'Entropie-Schwelle',
        'short': 'Entropie',
        'description': 'Entropie über Median = 1 (hohe Komplexität)'
    },
    'v6': {
        'name': 'Kompressibilität',
        'short': 'Kompress.',
        'description': 'Schlecht komprimierbar (hohe Komplexität) = 1'
    },
    'v7': {
        'name': 'Nachbar-Korrelation',
        'short': 'Nachbar',
        'description': 'Positiv korreliert mit Nachbar-Neuron = 1'
    }
}

# Aggregationsmethoden für Korrelation (nur die "einfachen")
AGGREGATION_METHODS = {
    'v0': {
        'name': 'Summe (Original)',
        'short': 'Summe',
        'description': 'Summe über alle Neuronen pro Iteration'
    },
    'v2': {
        'name': 'Skewness (Schiefe)',
        'short': 'Skewness',
        'description': 'Schiefe über alle Neuronen pro Iteration'
    },
    'v3': {
        'name': 'Median',
        'short': 'Median',
        'description': 'Median über alle Neuronen pro Iteration'
    }
}


class BitstringCalculator:
    """Berechnet Bitstrings mit verschiedenen Methoden"""
    
    @staticmethod
    def v0_summe(data):
        """Original: Summe→Vorzeichen"""
        sums = np.sum(data, axis=0)
        signs = np.sign(sums)
        signs[signs == 0] = 1
        return ''.join(['1' if s > 0 else '0' for s in signs])
    
    @staticmethod
    def v1_wendepunkt(data):
        """Wendepunkt-Dominanz: Mehr Aufwärts- als Abwärts-Wendepunkte = 1"""
        n_neurons = data.shape[1]
        bits = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            
            # Zähle Wendepunkte
            up_turns = 0
            down_turns = 0
            
            for i in range(1, len(neuron_data) - 1):
                prev_val = neuron_data[i-1]
                curr_val = neuron_data[i]
                next_val = neuron_data[i+1]
                
                # Aufwärts-Wendepunkt: war fallend, wird steigend
                if prev_val > curr_val and curr_val < next_val:
                    up_turns += 1
                # Abwärts-Wendepunkt: war steigend, wird fallend
                elif prev_val < curr_val and curr_val > next_val:
                    down_turns += 1
            
            bits.append('1' if up_turns >= down_turns else '0')
        
        return ''.join(bits)
    
    @staticmethod
    def v2_skewness(data):
        """Skewness: Rechtsschiefe = 1, Linksschiefe = 0"""
        n_neurons = data.shape[1]
        bits = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            skew = stats.skew(neuron_data)
            bits.append('1' if skew > 0 else '0')
        
        return ''.join(bits)
    
    @staticmethod
    def v3_median(data):
        """Median→Vorzeichen"""
        medians = np.median(data, axis=0)
        return ''.join(['1' if m > 0 else '0' for m in medians])
    
    @staticmethod
    def v4_fft_dominanz(data):
        """FFT-Dominanz: Dominante Frequenz über Median = 1"""
        n_neurons = data.shape[1]
        dominant_freqs = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            
            # FFT berechnen
            fft_vals = np.fft.fft(neuron_data)
            fft_magnitudes = np.abs(fft_vals)
            
            # Nur positive Frequenzen (erste Hälfte, ohne DC-Komponente)
            n = len(fft_magnitudes)
            positive_freqs = fft_magnitudes[1:n//2]
            
            if len(positive_freqs) > 0:
                # Index der dominanten Frequenz
                dominant_idx = np.argmax(positive_freqs) + 1
                dominant_freqs.append(dominant_idx)
            else:
                dominant_freqs.append(0)
        
        # Median der dominanten Frequenzen
        median_freq = np.median(dominant_freqs)
        
        # Über Median = 1
        return ''.join(['1' if f > median_freq else '0' for f in dominant_freqs])
    
    @staticmethod
    def v5_entropie(data):
        """Entropie-Schwelle: Hohe Entropie = 1"""
        n_neurons = data.shape[1]
        entropies = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            
            # Diskretisiere für Entropie-Berechnung (10 Bins)
            hist, _ = np.histogram(neuron_data, bins=10)
            hist = hist / hist.sum()  # Normalisieren
            
            # Shannon-Entropie
            entropy = 0
            for p in hist:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            entropies.append(entropy)
        
        # Median als Schwelle
        median_entropy = np.median(entropies)
        
        return ''.join(['1' if e > median_entropy else '0' for e in entropies])
    
    @staticmethod
    def v6_kompressibilitaet(data):
        """Kompressibilität: Schlecht komprimierbar = 1"""
        n_neurons = data.shape[1]
        compression_ratios = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            
            # Konvertiere zu Bytes
            data_bytes = neuron_data.tobytes()
            compressed = zlib.compress(data_bytes, level=9)
            
            ratio = len(compressed) / len(data_bytes)
            compression_ratios.append(ratio)
        
        # Median als Schwelle
        median_ratio = np.median(compression_ratios)
        
        # Hohe Ratio = schlecht komprimierbar = 1
        return ''.join(['1' if r > median_ratio else '0' for r in compression_ratios])
    
    @staticmethod
    def v7_nachbar_korrelation(data):
        """Nachbar-Korrelation: Positiv korreliert mit Nachbar = 1"""
        n_neurons = data.shape[1]
        bits = []
        
        for col in range(n_neurons):
            neuron_data = data[:, col]
            
            # Nachbar-Index (zyklisch)
            neighbor_col = (col + 1) % n_neurons
            neighbor_data = data[:, neighbor_col]
            
            # Pearson-Korrelation
            if np.std(neuron_data) > 0 and np.std(neighbor_data) > 0:
                corr = np.corrcoef(neuron_data, neighbor_data)[0, 1]
            else:
                corr = 0
            
            bits.append('1' if corr > 0 else '0')
        
        return ''.join(bits)
    
    @classmethod
    def calculate_all(cls, data):
        """Berechnet alle Bitstrings"""
        return {
            'v0': cls.v0_summe(data),
            'v1': cls.v1_wendepunkt(data),
            'v2': cls.v2_skewness(data),
            'v3': cls.v3_median(data),
            'v4': cls.v4_fft_dominanz(data),
            'v5': cls.v5_entropie(data),
            'v6': cls.v6_kompressibilitaet(data),
            'v7': cls.v7_nachbar_korrelation(data)
        }
    
    @classmethod
    def calculate(cls, data, method):
        """Berechnet Bitstring für eine Methode"""
        methods = {
            'v0': cls.v0_summe,
            'v1': cls.v1_wendepunkt,
            'v2': cls.v2_skewness,
            'v3': cls.v3_median,
            'v4': cls.v4_fft_dominanz,
            'v5': cls.v5_entropie,
            'v6': cls.v6_kompressibilitaet,
            'v7': cls.v7_nachbar_korrelation
        }
        return methods[method](data)


class AggregationCalculator:
    """Berechnet Aggregationswerte für Korrelation"""
    
    @staticmethod
    def v0_summe(data):
        """Summe über alle Neuronen pro Iteration (Original: mean)"""
        return np.mean(data, axis=1)
    
    @staticmethod
    def v2_skewness(data):
        """Skewness über alle Neuronen pro Iteration"""
        n_iterations = data.shape[0]
        result = np.zeros(n_iterations)
        for i in range(n_iterations):
            result[i] = stats.skew(data[i, :])
        return result
    
    @staticmethod
    def v3_median(data):
        """Median über alle Neuronen pro Iteration"""
        return np.median(data, axis=1)
    
    @classmethod
    def calculate(cls, data, method):
        """Berechnet Aggregation für eine Methode"""
        methods = {
            'v0': cls.v0_summe,
            'v2': cls.v2_skewness,
            'v3': cls.v3_median
        }
        return methods[method](data)


class LayerAnalyzer:
    """Kernlogik für Layer-Analyse"""
    
    def __init__(self):
        self.layers = {}  # name -> {'data': array, 'bitstrings': {method: str}, ...}
        self.layer_order = []
        self.correlation_matrices = {}  # method -> matrix
        self.sign_matrices = {}  # method -> matrix
        self.clusters_per_method = {}  # method -> clusters
        self.cluster_sequences = {}  # method -> sequence
        self.combined_bitstrings = {}  # method -> combined bitstring
        self.harmonized_iterations = None
        self.current_aggregation_method = 'v0'  # Default
    
    def load_layer(self, filepath, name=None):
        """Lädt einen Layer aus CSV"""
        if name is None:
            name = Path(filepath).stem
        
        df = pd.read_csv(filepath, header=None)
        data = df.values
        
        # Berechne alle Bitstrings
        bitstrings = BitstringCalculator.calculate_all(data)
        
        # Berechne alle Aggregationen für Korrelation
        aggregations = {}
        for method in AGGREGATION_METHODS.keys():
            aggregations[method] = AggregationCalculator.calculate(data, method)
        
        self.layers[name] = {
            'data': data,
            'bitstrings': bitstrings,
            'aggregations': aggregations,
            'n_neurons': data.shape[1],
            'n_iterations': data.shape[0],
            'n_iterations_original': data.shape[0]
        }
        
        return name
    
    def set_layer_order(self, order):
        """Setzt die Reihenfolge der Layer"""
        self.layer_order = order
        self.harmonize_iterations()
    
    def harmonize_iterations(self):
        """Kürzt alle Layer auf die minimale gemeinsame Iterationszahl"""
        if not self.layer_order:
            return
        
        min_iterations = min(self.layers[name]['n_iterations'] for name in self.layer_order)
        max_iterations = max(self.layers[name]['n_iterations'] for name in self.layer_order)
        
        self.harmonized_iterations = min_iterations
        
        if min_iterations < max_iterations:
            print(f"HINWEIS: Iterationen unterschiedlich ({min_iterations} - {max_iterations})")
            print(f"         Alle Layer werden auf {min_iterations} Iterationen gekürzt")
            
            for name in self.layer_order:
                layer = self.layers[name]
                if layer['n_iterations'] > min_iterations:
                    data = layer['data'][:min_iterations, :]
                    
                    # Neu berechnen
                    bitstrings = BitstringCalculator.calculate_all(data)
                    aggregations = {}
                    for method in AGGREGATION_METHODS.keys():
                        aggregations[method] = AggregationCalculator.calculate(data, method)
                    
                    layer['data'] = data
                    layer['bitstrings'] = bitstrings
                    layer['aggregations'] = aggregations
                    layer['n_iterations'] = min_iterations
    
    def compute_correlation_matrix(self, method='v0'):
        """Berechnet Pearson-Korrelation zwischen Layer-Zeitreihen für eine Methode"""
        if not self.layer_order:
            return None
        
        n = len(self.layer_order)
        matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(self.layer_order):
            for j, name2 in enumerate(self.layer_order):
                agg1 = self.layers[name1]['aggregations'][method]
                agg2 = self.layers[name2]['aggregations'][method]
                corr = np.corrcoef(agg1, agg2)[0, 1]
                if np.isnan(corr):
                    corr = 0
                matrix[i, j] = corr
        
        self.correlation_matrices[method] = matrix
        self.sign_matrices[method] = np.sign(matrix)
        
        return matrix
    
    def compute_all_correlations(self):
        """Berechnet Korrelationen für alle Aggregationsmethoden"""
        for method in AGGREGATION_METHODS.keys():
            self.compute_correlation_matrix(method)
    
    def identify_clusters(self, method='v0'):
        """Identifiziert Cluster basierend auf Korrelationsmustern für eine Methode"""
        if method not in self.sign_matrices:
            self.compute_correlation_matrix(method)
        
        sign_matrix = self.sign_matrices[method]
        n = len(self.layer_order)
        
        row_patterns = {}
        for i, name in enumerate(self.layer_order):
            pattern = ''.join(['+' if sign_matrix[i, j] >= 0 else '-' for j in range(n)])
            if pattern not in row_patterns:
                row_patterns[pattern] = []
            row_patterns[pattern].append(name)
        
        greek = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
                 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
        
        clusters = {}
        cluster_idx = 0
        layer_to_cluster = {}
        
        for pattern, layers in row_patterns.items():
            if cluster_idx < len(greek):
                cluster_name = greek[cluster_idx]
            else:
                cluster_name = f"C{cluster_idx}"
            
            clusters[cluster_name] = {
                'pattern': pattern,
                'layers': layers,
                'count': len(layers)
            }
            
            for layer in layers:
                layer_to_cluster[layer] = cluster_name
            
            cluster_idx += 1
        
        cluster_sequence = ''.join([layer_to_cluster[name] for name in self.layer_order])
        
        self.clusters_per_method[method] = clusters
        self.cluster_sequences[method] = cluster_sequence
        
        return clusters
    
    def identify_all_clusters(self):
        """Identifiziert Cluster für alle Aggregationsmethoden"""
        for method in AGGREGATION_METHODS.keys():
            self.identify_clusters(method)
    
    def compute_hamming_distances(self, method='v0'):
        """Berechnet Hamming-Distanzen zwischen Cluster-Patterns für eine Methode"""
        if method not in self.clusters_per_method:
            return {}
        
        clusters = self.clusters_per_method[method]
        distances = {}
        cluster_names = list(clusters.keys())
        
        for i, c1 in enumerate(cluster_names):
            for c2 in cluster_names[i+1:]:
                p1 = clusters[c1]['pattern']
                p2 = clusters[c2]['pattern']
                dist = sum(a != b for a, b in zip(p1, p2))
                distances[f"{c1}↔{c2}"] = dist
        
        return distances
    
    def extract_combined_bitstrings(self):
        """Extrahiert kombinierte Bitstrings für alle Methoden"""
        if not self.layer_order:
            return {}
        
        self.combined_bitstrings = {}
        
        for method in BITSTRING_METHODS.keys():
            combined = ''.join([self.layers[name]['bitstrings'][method] 
                               for name in self.layer_order])
            self.combined_bitstrings[method] = combined
        
        return self.combined_bitstrings
    
    def analyze_bitstring(self, bitstring):
        """Analysiert einen Bitstring"""
        if not bitstring:
            return {}
        
        n = len(bitstring)
        ones = bitstring.count('1')
        zeros = bitstring.count('0')
        
        # Entropie
        entropy = 0
        for count in [ones, zeros]:
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        
        # Kompression
        original = bitstring.encode('utf-8')
        compressed = zlib.compress(original, level=9)
        compression_ratio = len(compressed) / len(original)
        
        # Run-Length
        runs = [(bit, len(list(group))) for bit, group in groupby(bitstring)]
        avg_run_length = n / len(runs) if runs else 0
        
        return {
            'length': n,
            'ones': ones,
            'zeros': zeros,
            'balance': ones / n if n > 0 else 0,
            'entropy': entropy,
            'compression_ratio': compression_ratio,
            'n_runs': len(runs),
            'avg_run_length': avg_run_length
        }
    
    def get_method_summary(self, bitstring_method):
        """Erstellt Zusammenfassung für eine Bitstring-Methode (v4-kompatibles Format!)"""
        if not self.layer_order or bitstring_method not in self.combined_bitstrings:
            return "Keine Daten."
        
        method_info = BITSTRING_METHODS[bitstring_method]
        
        summary = []
        summary.append("=" * 70)
        summary.append(f"BITSTRING-ANALYSE: {bitstring_method} - {method_info['name']}")
        summary.append("=" * 70)
        summary.append(f"Beschreibung: {method_info['description']}")
        summary.append(f"Analysiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Anzahl Layer: {len(self.layer_order)}")
        
        if self.harmonized_iterations is not None:
            summary.append(f"Iterationen (harmonisiert): {self.harmonized_iterations}")
        
        summary.append("")
        summary.append("-" * 70)
        summary.append("LAYER-BITSTRINGS")
        summary.append("-" * 70)
        
        total_neurons = 0
        for name in self.layer_order:
            layer = self.layers[name]
            bitstring = layer['bitstrings'][bitstring_method]
            total_neurons += layer['n_neurons']
            # v4-kompatibles Format!
            summary.append(f"{name}: {layer['n_neurons']} Neuronen, Bitstring: {bitstring}")
        
        summary.append(f"\nGesamt: {total_neurons} Neuronen")
        
        # Kombinierter Bitstring
        combined = self.combined_bitstrings[bitstring_method]
        analysis = self.analyze_bitstring(combined)
        
        summary.append("")
        summary.append("-" * 70)
        summary.append("KOMBINIERTER BITSTRING")
        summary.append("-" * 70)
        
        # Zeige Bitstring in Blöcken von 64
        for i in range(0, len(combined), 64):
            summary.append(f"  {combined[i:i+64]}")
        
        summary.append("")
        summary.append("STATISTIKEN:")
        summary.append(f"  Länge: {analysis['length']} Bits")
        summary.append(f"  Einsen: {analysis['ones']} ({100*analysis['balance']:.1f}%)")
        summary.append(f"  Nullen: {analysis['zeros']} ({100*(1-analysis['balance']):.1f}%)")
        summary.append(f"  Shannon-Entropie: {analysis['entropy']:.4f} bits/symbol")
        summary.append(f"  Kompressionsrate: {analysis['compression_ratio']:.4f}")
        summary.append(f"  Durchschn. Run-Länge: {analysis['avg_run_length']:.2f}")
        
        summary.append("")
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def get_summary(self, aggregation_method='v0'):
        """Erstellt Hauptzusammenfassung (v4-kompatibles Format!)"""
        if not self.layer_order:
            return "Keine Layer geladen."
        
        agg_info = AGGREGATION_METHODS.get(aggregation_method, AGGREGATION_METHODS['v0'])
        
        summary = []
        summary.append("=" * 70)
        summary.append("GCIS LAYER ANALYSE - ZUSAMMENFASSUNG")
        summary.append("=" * 70)
        summary.append(f"Analysiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Anzahl Layer: {len(self.layer_order)}")
        summary.append(f"Aggregationsmethode: {aggregation_method} - {agg_info['name']}")
        
        if self.harmonized_iterations is not None:
            original_iterations = [self.layers[name].get('n_iterations_original',
                                   self.layers[name]['n_iterations'])
                                   for name in self.layer_order]
            if max(original_iterations) > self.harmonized_iterations:
                summary.append("")
                summary.append(f"*** ITERATIONS-HARMONISIERUNG ***")
                summary.append(f"Original-Iterationen: {min(original_iterations)} - {max(original_iterations)}")
                summary.append(f"Harmonisiert auf: {self.harmonized_iterations} Iterationen")
        
        summary.append("")
        summary.append("-" * 70)
        summary.append("LAYER-DETAILS (in Analysereihenfolge)")
        summary.append("-" * 70)
        
        total_neurons = 0
        for name in self.layer_order:
            layer = self.layers[name]
            total_neurons += layer['n_neurons']
            # v4-kompatibles Format!
            summary.append(f"{name}: {layer['n_neurons']} Neuronen, "
                          f"Bitstring: {layer['bitstrings']['v0']}")
        
        summary.append(f"\nGesamt: {total_neurons} Neuronen")
        
        # Korrelationsmatrix
        if aggregation_method in self.correlation_matrices:
            corr_matrix = self.correlation_matrices[aggregation_method]
            summary.append("")
            summary.append("-" * 70)
            summary.append(f"PEARSON-KORRELATIONSMATRIX ({agg_info['short']})")
            summary.append("-" * 70)
            
            header = "         " + "".join([f"{name:>10}" for name in self.layer_order])
            summary.append(header)
            
            for i, name in enumerate(self.layer_order):
                row = f"{name:>8} "
                for j in range(len(self.layer_order)):
                    val = corr_matrix[i, j]
                    row += f"{val:>10.4f}"
                summary.append(row)
        
        # Cluster
        if aggregation_method in self.clusters_per_method:
            clusters = self.clusters_per_method[aggregation_method]
            cluster_sequence = self.cluster_sequences[aggregation_method]
            
            summary.append("")
            summary.append("-" * 70)
            summary.append(f"CLUSTER-ANALYSE ({agg_info['short']})")
            summary.append("-" * 70)
            summary.append(f"Anzahl eindeutiger Cluster: {len(clusters)}")
            summary.append("")
            
            for cname, cdata in clusters.items():
                summary.append(f"Cluster {cname}:")
                summary.append(f"  Pattern: {cdata['pattern']}")
                summary.append(f"  Layer: {', '.join(cdata['layers'])}")
            
            summary.append("")
            summary.append(f"Cluster-Sequenz ('Genom'): {cluster_sequence}")
            
            # Hamming-Distanzen
            distances = self.compute_hamming_distances(aggregation_method)
            if distances:
                summary.append("")
                summary.append("Hamming-Distanzen:")
                for pair, dist in sorted(distances.items()):
                    summary.append(f"  {pair}: {dist}")
        
        # Bitstring-Vergleich
        if self.combined_bitstrings:
            summary.append("")
            summary.append("-" * 70)
            summary.append("BITSTRING-METHODEN ÜBERSICHT")
            summary.append("-" * 70)
            
            for method, info in BITSTRING_METHODS.items():
                if method in self.combined_bitstrings:
                    bs = self.combined_bitstrings[method]
                    analysis = self.analyze_bitstring(bs)
                    summary.append(f"{method} ({info['short']}): "
                                  f"{analysis['length']} Bits, "
                                  f"Balance: {100*analysis['balance']:.1f}% Einsen, "
                                  f"Entropie: {analysis['entropy']:.3f}")
        
        summary.append("")
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def export_method_json(self, method, filepath):
        """Exportiert Daten einer Methode als JSON"""
        if method not in self.combined_bitstrings:
            return None
        
        method_info = BITSTRING_METHODS[method]
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'method_name': method_info['name'],
            'method_description': method_info['description'],
            'layer_order': self.layer_order,
            'harmonized_iterations': self.harmonized_iterations,
            'layers': {},
            'combined_bitstring': self.combined_bitstrings[method],
            'bitstring_analysis': self.analyze_bitstring(self.combined_bitstrings[method])
        }
        
        for name in self.layer_order:
            layer = self.layers[name]
            export_data['layers'][name] = {
                'n_neurons': layer['n_neurons'],
                'n_iterations': layer['n_iterations'],
                'bitstring': layer['bitstrings'][method]
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_json(self, filepath, aggregation_method='v0'):
        """Exportiert alle Daten als JSON"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'aggregation_method': aggregation_method,
            'layer_order': self.layer_order,
            'harmonized_iterations': self.harmonized_iterations,
            'layers': {},
            'correlation_matrix': self.correlation_matrices.get(aggregation_method, np.array([])).tolist(),
            'sign_matrix': self.sign_matrices.get(aggregation_method, np.array([])).tolist(),
            'clusters': self.clusters_per_method.get(aggregation_method, {}),
            'cluster_sequence': self.cluster_sequences.get(aggregation_method, ""),
            'combined_bitstrings': self.combined_bitstrings,
            'bitstring_analyses': {}
        }
        
        for method, bs in self.combined_bitstrings.items():
            export_data['bitstring_analyses'][method] = self.analyze_bitstring(bs)
        
        for name in self.layer_order:
            layer = self.layers[name]
            export_data['layers'][name] = {
                'n_neurons': layer['n_neurons'],
                'n_iterations': layer['n_iterations'],
                'n_iterations_original': layer.get('n_iterations_original', layer['n_iterations']),
                'bitstrings': layer['bitstrings']
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filepath


def natural_sort_key(name):
    """Sortiert natürlich: zfa1, zfa2, ... zfa10, zfa11"""
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def sortiere_layer_namen(layer_namen):
    """Sortiert Layer nach Präfix und dann numerisch."""
    def sort_key(name):
        match = re.match(r'^([a-zA-Z]+)(\d*)', name)
        if match:
            prefix = match.group(1).lower()
            num_str = match.group(2)
            num = int(num_str) if num_str else 0
            return (prefix, num, name)
        return (name.lower(), 0, name)
    
    return sorted(layer_namen, key=sort_key)


def get_prefix(name):
    """Extrahiert Buchstaben-Präfix aus Layername"""
    match = re.match(r'^([a-zA-Z]+)', name)
    return match.group(1).lower() if match else "other"


class AnalyzerGUI:
    """GUI für Layer-Analyse mit Checkbox-Auswahl"""
    
    def __init__(self):
        if not HAS_TKINTER:
            raise ImportError("tkinter ist nicht installiert.")
        
        self.root = tk.Tk()
        self.root.title("GCIS Layer Analyzer v6.0 - Korrelation pro Methode")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLORS['white'])
        
        self.analyzer = LayerAnalyzer()
        
        self.loaded_files = {}
        self.name_to_filepath = {}
        self.alle_daten = {}
        
        self.layer_checkboxen = {}
        self.layer_labels = {}
        self.cluster_vars = {}
        self.layer_reihenfolge = {}
        self.reihenfolge_counter = 0
        
        self.setup_gui()
    
    def setup_gui(self):
        """Erstellt GUI"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Layer-Auswahl
        self.tab_selection = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_selection, text="1. Layer-Auswahl")
        self.setup_selection_tab()
        
        # Tab 2: Hauptanalyse
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="2. Analyse")
        self.setup_analysis_tab()
        
        # Tab 3: Bitstring-Methoden
        self.tab_bitstrings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bitstrings, text="3. Bitstring-Methoden")
        self.setup_bitstrings_tab()
        
        # Tab 4: Visualisierung
        self.tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz, text="4. Visualisierung")
        self.setup_viz_tab()
    
    def setup_selection_tab(self):
        """Setup für Layer-Auswahl"""
        main_frame = ttk.Frame(self.tab_selection)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Obere Leiste
        top_frame = tk.Frame(main_frame, bg=COLORS['white'])
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.quell_label = tk.Label(top_frame, text="Quellordner: Nicht gewählt",
                                    bg=COLORS['white'], fg=COLORS['dark'])
        self.quell_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="📂 Ordner wählen",
                  command=self.waehle_quellordner,
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="🗑 Auswahl zurücksetzen",
                  command=self.reset_auswahl,
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=20)
        
        # Hauptbereich: 2 Spalten
        columns_frame = tk.Frame(main_frame, bg=COLORS['white'])
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linke Seite: Checkboxen
        left_frame = tk.LabelFrame(columns_frame, text="LAYER AUSWAHL",
                                   bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                   font=('Verdana', 10, 'bold'))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        canvas = tk.Canvas(left_frame, bg=COLORS['white'])
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.layer_scroll_frame = tk.Frame(canvas, bg=COLORS['white'])
        
        self.layer_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.layer_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Rechte Seite: Finale Reihenfolge
        right_frame = tk.LabelFrame(columns_frame, text="FINALE ANALYSE-REIHENFOLGE",
                                    bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                    font=('Verdana', 10, 'bold'))
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.final_info_label = tk.Label(right_frame, text="0 Layer ausgewählt",
                                          bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                          font=('Verdana', 10, 'bold'))
        self.final_info_label.pack(padx=5, pady=5)
        
        final_inner = ttk.Frame(right_frame)
        final_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.final_listbox = tk.Listbox(final_inner, font=('Courier', 10))
        self.final_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        final_scroll = ttk.Scrollbar(final_inner, orient=tk.VERTICAL)
        final_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.final_listbox.config(yscrollcommand=final_scroll.set)
        final_scroll.config(command=self.final_listbox.yview)
        
        tk.Button(right_frame, text="▶ ANALYSE STARTEN",
                  command=self.run_analysis,
                  bg=COLORS['gold'], fg=COLORS['dark'],
                  font=('Verdana', 12, 'bold')).pack(pady=10)
    
    def setup_analysis_tab(self):
        """Setup für Analyse-Tab mit Aggregations-Dropdown"""
        main_frame = ttk.Frame(self.tab_analysis)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Obere Leiste: Aggregationsmethode auswählen
        top_frame = tk.Frame(main_frame, bg=COLORS['white'])
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(top_frame, text="Aggregationsmethode für Korrelation/Cluster:",
                 bg=COLORS['white'], fg=COLORS['dark'],
                 font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.aggregation_var = tk.StringVar(value='v0')
        self.aggregation_combo = ttk.Combobox(top_frame, textvariable=self.aggregation_var,
                                               state='readonly', width=30)
        agg_options = [f"{k}: {v['name']}" for k, v in AGGREGATION_METHODS.items()]
        self.aggregation_combo['values'] = agg_options
        self.aggregation_combo.current(0)
        self.aggregation_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="Aktualisieren",
                  command=self.update_analysis_display,
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=10)
        
        # Textanzeige
        self.result_text = scrolledtext.ScrolledText(main_frame,
                                                      wrap=tk.WORD,
                                                      font=('Courier', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export-Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Als Text speichern",
                   command=self.export_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Als JSON speichern",
                   command=self.export_json).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Bitstring v0 kopieren",
                   command=self.copy_bitstring).pack(side=tk.LEFT, padx=5)
    
    def setup_bitstrings_tab(self):
        """Setup für Bitstring-Methoden Tab"""
        main_frame = ttk.Frame(self.tab_bitstrings)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Obere Leiste: Methoden-Auswahl
        top_frame = tk.Frame(main_frame, bg=COLORS['white'])
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(top_frame, text="Bitstring-Methode auswählen:",
                 bg=COLORS['white'], fg=COLORS['dark'],
                 font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.method_var = tk.StringVar(value='v0')
        self.method_combo = ttk.Combobox(top_frame, textvariable=self.method_var,
                                          state='readonly', width=40)
        method_options = [f"{k}: {v['name']}" for k, v in BITSTRING_METHODS.items()]
        self.method_combo['values'] = method_options
        self.method_combo.current(0)
        self.method_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="Anzeigen",
                  command=self.show_method_analysis,
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=10)
        
        # Textanzeige
        self.method_text = scrolledtext.ScrolledText(main_frame,
                                                      wrap=tk.WORD,
                                                      font=('Courier', 10))
        self.method_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export-Buttons
        btn_frame = tk.LabelFrame(main_frame, text="Export (gewählte Methode)",
                                   bg=COLORS['light_gray'], fg=COLORS['gold_dark'])
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Als TXT speichern",
                   command=self.export_method_txt).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(btn_frame, text="Als JSON speichern",
                   command=self.export_method_json).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(btn_frame, text="Bitstring kopieren",
                   command=self.copy_method_bitstring).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Alle Methoden exportieren
        all_frame = tk.LabelFrame(main_frame, text="Alle Methoden exportieren",
                                   bg=COLORS['light_gray'], fg=COLORS['gold_dark'])
        all_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(all_frame, text="Alle als TXT exportieren",
                   command=self.export_all_methods_txt).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(all_frame, text="Alle als JSON exportieren",
                   command=self.export_all_methods_json).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_viz_tab(self):
        """Setup für Visualisierungs-Tab mit Aggregations-Dropdown"""
        if not HAS_MATPLOTLIB:
            ttk.Label(self.tab_viz,
                      text="Matplotlib nicht installiert.",
                      font=('Arial', 14)).pack(expand=True)
            return
        
        # Obere Leiste: Aggregationsmethode auswählen
        top_frame = tk.Frame(self.tab_viz, bg=COLORS['white'])
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(top_frame, text="Aggregationsmethode:",
                 bg=COLORS['white'], fg=COLORS['dark'],
                 font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.viz_aggregation_var = tk.StringVar(value='v0')
        self.viz_aggregation_combo = ttk.Combobox(top_frame, textvariable=self.viz_aggregation_var,
                                                   state='readonly', width=30)
        agg_options = [f"{k}: {v['name']}" for k, v in AGGREGATION_METHODS.items()]
        self.viz_aggregation_combo['values'] = agg_options
        self.viz_aggregation_combo.current(0)
        self.viz_aggregation_combo.pack(side=tk.LEFT, padx=5)
        
        self.fig_frame = ttk.Frame(self.tab_viz)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(self.tab_viz)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Korrelationsmatrix",
                   command=self.plot_correlation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cluster-Sequenz",
                   command=self.plot_clusters).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Hamming-Distanzen",
                   command=self.plot_hamming).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Bild speichern",
                   command=self.save_figure).pack(side=tk.LEFT, padx=5)
        
        self.current_fig = None
    
    # ==================== ORDNER & DATEN ====================
    
    def waehle_quellordner(self):
        """Wählt Quellordner und lädt CSVs"""
        ordner = filedialog.askdirectory(title="Quellordner wählen")
        if ordner:
            self.quell_label.config(text=f"Quellordner: {ordner}")
            
            self.loaded_files = {}
            self.name_to_filepath = {}
            
            for filepath in sorted(Path(ordner).glob("*.csv"), key=lambda p: natural_sort_key(p.stem)):
                name = filepath.stem
                filepath_str = str(filepath)
                self.loaded_files[filepath_str] = name
                self.name_to_filepath[name] = filepath_str
            
            self.aktualisiere_layer_checkboxen()
            messagebox.showinfo("Info", f"{len(self.loaded_files)} Layer geladen.")
    
    def aktualisiere_layer_checkboxen(self):
        """Erstellt Checkboxen gruppiert nach Präfix"""
        for widget in self.layer_scroll_frame.winfo_children():
            widget.destroy()
        
        self.layer_checkboxen = {}
        self.layer_labels = {}
        self.cluster_vars = {}
        self.layer_reihenfolge = {}
        self.reihenfolge_counter = 0
        
        layer_namen = sortiere_layer_namen(list(self.name_to_filepath.keys()))
        
        cluster = {}
        for name in layer_namen:
            prefix = get_prefix(name)
            if prefix not in cluster:
                cluster[prefix] = []
            cluster[prefix].append(name)
        
        row = 0
        for prefix in sorted(cluster.keys()):
            header_frame = tk.Frame(self.layer_scroll_frame, bg=COLORS['gold_light'])
            header_frame.grid(row=row, column=0, columnspan=4, sticky='ew', pady=(10, 2), padx=2)
            
            cluster_var = tk.BooleanVar()
            self.cluster_vars[prefix] = cluster_var
            
            cluster_cb = tk.Checkbutton(header_frame,
                                        text=f"▶ {prefix.upper()} ({len(cluster[prefix])} Layer)",
                                        variable=cluster_var,
                                        bg=COLORS['gold_light'],
                                        font=('Verdana', 9, 'bold'),
                                        command=lambda p=prefix: self.cluster_auswahl_geaendert(p))
            cluster_cb.pack(side=tk.LEFT, padx=5)
            row += 1
            
            col = 0
            for layer_name in cluster[prefix]:
                frame = tk.Frame(self.layer_scroll_frame, bg=COLORS['white'])
                frame.grid(row=row, column=col, sticky='w', padx=15, pady=1)
                
                var = tk.BooleanVar()
                
                cb = tk.Checkbutton(frame, text=layer_name,
                                    variable=var, bg=COLORS['white'],
                                    command=lambda n=layer_name: self.layer_auswahl_geaendert(n))
                cb.pack(side=tk.LEFT)
                
                label = tk.Label(frame, text="", bg=COLORS['white'],
                                 fg=COLORS['gold_dark'], font=('Verdana', 9, 'bold'))
                label.pack(side=tk.LEFT, padx=(5, 0))
                
                self.layer_checkboxen[layer_name] = var
                self.layer_labels[layer_name] = label
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
            
            if col != 0:
                row += 1
        
        self.update_final_order()
    
    def cluster_auswahl_geaendert(self, prefix):
        """Wählt alle Layer eines Clusters an oder ab"""
        cluster_aktiv = self.cluster_vars[prefix].get()
        
        for layer_name, var in self.layer_checkboxen.items():
            layer_prefix = get_prefix(layer_name)
            
            if layer_prefix == prefix:
                if cluster_aktiv and not var.get():
                    var.set(True)
                    self.reihenfolge_counter += 1
                    self.layer_reihenfolge[layer_name] = self.reihenfolge_counter
                    self.layer_labels[layer_name].config(text=f"({self.reihenfolge_counter})")
                elif not cluster_aktiv and var.get():
                    var.set(False)
                    if layer_name in self.layer_reihenfolge:
                        del self.layer_reihenfolge[layer_name]
                    self.layer_labels[layer_name].config(text="")
        
        self.update_final_order()
    
    def layer_auswahl_geaendert(self, layer_name):
        """Wird aufgerufen wenn einzelner Layer an/abgewählt wird"""
        if self.layer_checkboxen[layer_name].get():
            self.reihenfolge_counter += 1
            self.layer_reihenfolge[layer_name] = self.reihenfolge_counter
            self.layer_labels[layer_name].config(text=f"({self.reihenfolge_counter})")
        else:
            if layer_name in self.layer_reihenfolge:
                del self.layer_reihenfolge[layer_name]
            self.layer_labels[layer_name].config(text="")
        
        self.update_final_order()
    
    def reset_auswahl(self):
        """Setzt alle Auswahlen zurück"""
        for name, var in self.layer_checkboxen.items():
            var.set(False)
            self.layer_labels[name].config(text="")
        
        for prefix, var in self.cluster_vars.items():
            var.set(False)
        
        self.layer_reihenfolge = {}
        self.reihenfolge_counter = 0
        self.update_final_order()
    
    def update_final_order(self):
        """Aktualisiert die finale Reihenfolge-Anzeige"""
        self.final_listbox.delete(0, tk.END)
        
        ausgewaehlte = [(name, pos) for name, pos in self.layer_reihenfolge.items()]
        ausgewaehlte.sort(key=lambda x: x[1])
        
        for name, pos in ausgewaehlte:
            self.final_listbox.insert(tk.END, f"({pos}) {name}")
        
        self.final_info_label.config(text=f"{len(ausgewaehlte)} Layer ausgewählt")
        
        return [name for name, pos in ausgewaehlte]
    
    def get_final_order(self):
        """Gibt die finale Reihenfolge zurück"""
        ausgewaehlte = [(name, pos) for name, pos in self.layer_reihenfolge.items()]
        ausgewaehlte.sort(key=lambda x: x[1])
        return [name for name, pos in ausgewaehlte]
    
    # ==================== ANALYSE ====================
    
    def get_selected_aggregation(self):
        """Gibt den ausgewählten Aggregations-Code zurück"""
        selection = self.aggregation_var.get()
        return selection.split(':')[0]
    
    def get_viz_aggregation(self):
        """Gibt den ausgewählten Aggregations-Code für Visualisierung zurück"""
        selection = self.viz_aggregation_var.get()
        return selection.split(':')[0]
    
    def run_analysis(self):
        """Führt die Analyse durch"""
        final_order = self.get_final_order()
        
        if not final_order:
            messagebox.showwarning("Warnung", "Keine Layer ausgewählt!")
            return
        
        self.analyzer = LayerAnalyzer()
        
        for name in final_order:
            filepath = self.name_to_filepath.get(name)
            if filepath:
                try:
                    self.analyzer.load_layer(filepath, name)
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Laden von {name}:\n{e}")
                    return
        
        self.analyzer.set_layer_order(final_order)
        
        # Berechne alle Korrelationen und Cluster
        self.analyzer.compute_all_correlations()
        self.analyzer.identify_all_clusters()
        self.analyzer.extract_combined_bitstrings()
        
        # Zeige Analyse mit Default-Methode v0
        self.update_analysis_display()
        
        self.notebook.select(self.tab_analysis)
        
        harmonized = self.analyzer.harmonized_iterations
        original_iters = [self.analyzer.layers[n].get('n_iterations_original',
                         self.analyzer.layers[n]['n_iterations'])
                         for n in final_order]
        
        if max(original_iters) > harmonized:
            messagebox.showinfo("Analyse abgeschlossen",
                f"{len(final_order)} Layer analysiert.\n\n"
                f"⚠️ Iterations-Harmonisierung:\n"
                f"Original: {min(original_iters)} - {max(original_iters)}\n"
                f"Harmonisiert: {harmonized}\n\n"
                f"3 Aggregationsmethoden für Korrelation verfügbar.")
        else:
            messagebox.showinfo("Erfolg", f"Analyse mit {len(final_order)} Layern abgeschlossen!\n"
                                          f"8 Bitstring-Methoden berechnet.\n"
                                          f"3 Aggregationsmethoden für Korrelation verfügbar.")
    
    def update_analysis_display(self):
        """Aktualisiert die Analyse-Anzeige basierend auf gewählter Aggregationsmethode"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        method = self.get_selected_aggregation()
        summary = self.analyzer.get_summary(method)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, summary)
    
    # ==================== BITSTRING-METHODEN TAB ====================
    
    def get_selected_method(self):
        """Gibt den ausgewählten Methoden-Code zurück"""
        selection = self.method_var.get()
        return selection.split(':')[0]
    
    def show_method_analysis(self):
        """Zeigt Analyse für gewählte Methode"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        method = self.get_selected_method()
        summary = self.analyzer.get_method_summary(method)
        
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(tk.END, summary)
    
    def export_method_txt(self):
        """Exportiert gewählte Methode als TXT"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        method = self.get_selected_method()
        method_info = BITSTRING_METHODS[method]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"bitstring_{method}_{method_info['short']}.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            summary = self.analyzer.get_method_summary(method)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary)
            messagebox.showinfo("Erfolg", f"Gespeichert: {filepath}")
    
    def export_method_json(self):
        """Exportiert gewählte Methode als JSON"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        method = self.get_selected_method()
        method_info = BITSTRING_METHODS[method]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"bitstring_{method}_{method_info['short']}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.analyzer.export_method_json(method, filepath)
            messagebox.showinfo("Erfolg", f"Gespeichert: {filepath}")
    
    def copy_method_bitstring(self):
        """Kopiert Bitstring der gewählten Methode"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        method = self.get_selected_method()
        bitstring = self.analyzer.combined_bitstrings.get(method, "")
        
        if bitstring:
            self.root.clipboard_clear()
            self.root.clipboard_append(bitstring)
            messagebox.showinfo("Kopiert",
                                f"Bitstring {method} kopiert ({len(bitstring)} Bits)")
    
    def export_all_methods_txt(self):
        """Exportiert alle Methoden als TXT in einen Ordner"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        ordner = filedialog.askdirectory(title="Zielordner wählen")
        if ordner:
            for method, info in BITSTRING_METHODS.items():
                filepath = Path(ordner) / f"bitstring_{method}_{info['short']}.txt"
                summary = self.analyzer.get_method_summary(method)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(summary)
            
            messagebox.showinfo("Erfolg", f"8 TXT-Dateien in {ordner} gespeichert.")
    
    def export_all_methods_json(self):
        """Exportiert alle Methoden als JSON in einen Ordner"""
        if not self.analyzer.combined_bitstrings:
            messagebox.showwarning("Warnung", "Erst Analyse durchführen!")
            return
        
        ordner = filedialog.askdirectory(title="Zielordner wählen")
        if ordner:
            for method, info in BITSTRING_METHODS.items():
                filepath = Path(ordner) / f"bitstring_{method}_{info['short']}.json"
                self.analyzer.export_method_json(method, str(filepath))
            
            messagebox.showinfo("Erfolg", f"8 JSON-Dateien in {ordner} gespeichert.")
    
    # ==================== EXPORT (Hauptanalyse) ====================
    
    def export_text(self):
        """Exportiert als Text"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.result_text.get(1.0, tk.END))
            messagebox.showinfo("Erfolg", f"Gespeichert: {filepath}")
    
    def export_json(self):
        """Exportiert als JSON"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            method = self.get_selected_aggregation()
            self.analyzer.export_json(filepath, method)
            messagebox.showinfo("Erfolg", f"Gespeichert: {filepath}")
    
    def copy_bitstring(self):
        """Kopiert Bitstring v0 in Zwischenablage"""
        bitstring = self.analyzer.combined_bitstrings.get('v0', "")
        if bitstring:
            self.root.clipboard_clear()
            self.root.clipboard_append(bitstring)
            messagebox.showinfo("Kopiert",
                                f"Bitstring v0 kopiert ({len(bitstring)} Bits)")
    
    # ==================== VISUALISIERUNG ====================
    
    def clear_viz(self):
        """Löscht aktuelle Visualisierung"""
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
    
    def plot_correlation(self):
        """Plottet Korrelationsmatrix für gewählte Aggregationsmethode"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Warnung", "Matplotlib nicht installiert!")
            return
        
        method = self.get_viz_aggregation()
        
        if method not in self.analyzer.correlation_matrices:
            messagebox.showwarning("Warnung", "Keine Analyse-Daten! Erst Analyse durchführen.")
            return
        
        self.clear_viz()
        
        fig = Figure(figsize=(10, 8), facecolor=COLORS['bg_plot'])
        ax = fig.add_subplot(111, facecolor=COLORS['bg_plot'])
        
        corr_matrix = self.analyzer.correlation_matrices[method]
        method_info = AGGREGATION_METHODS[method]
        
        n = len(self.analyzer.layer_order)
        im = ax.imshow(corr_matrix, cmap='RdBu_r',
                       vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.analyzer.layer_order, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(self.analyzer.layer_order, fontsize=8)
        ax.set_title(f'Pearson-Korrelationsmatrix ({method_info["short"]})', 
                     fontsize=14, fontweight='bold')
        
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    val = corr_matrix[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=7)
        
        fig.colorbar(im, ax=ax, label='Korrelation')
        fig.tight_layout()
        
        self.current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_clusters(self):
        """Plottet Cluster-Sequenz für gewählte Aggregationsmethode"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Warnung", "Matplotlib nicht installiert!")
            return
        
        method = self.get_viz_aggregation()
        
        if method not in self.analyzer.clusters_per_method:
            messagebox.showwarning("Warnung", "Keine Cluster-Daten! Erst Analyse durchführen.")
            return
        
        self.clear_viz()
        
        fig = Figure(figsize=(12, 6), facecolor=COLORS['bg_plot'])
        ax = fig.add_subplot(111, facecolor=COLORS['bg_plot'])
        
        clusters = self.analyzer.clusters_per_method[method]
        cluster_sequence = self.analyzer.cluster_sequences[method]
        method_info = AGGREGATION_METHODS[method]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        cluster_colors = {name: colors[i] for i, name in enumerate(clusters.keys())}
        
        layer_to_cluster = {}
        for cname, cdata in clusters.items():
            for layer in cdata['layers']:
                layer_to_cluster[layer] = cname
        
        n = len(self.analyzer.layer_order)
        for i, name in enumerate(self.analyzer.layer_order):
            cluster = layer_to_cluster[name]
            color = cluster_colors[cluster]
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color,
                                        edgecolor='black', linewidth=0.5))
            ax.text(i + 0.5, 0.5, cluster, ha='center', va='center',
                    fontsize=10, fontweight='bold')
            ax.text(i + 0.5, -0.3, name, ha='center', va='top',
                    fontsize=8, rotation=45)
        
        ax.set_xlim(0, n)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Cluster-Sequenz ({method_info["short"]}): {cluster_sequence}',
                     fontsize=14, fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cluster_colors[name], edgecolor='black',
                                 label=f'{name}: {data["count"]} Layer')
                           for name, data in clusters.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        
        self.current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_hamming(self):
        """Plottet Hamming-Distanz-Matrix für gewählte Aggregationsmethode"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Warnung", "Matplotlib nicht installiert!")
            return
        
        method = self.get_viz_aggregation()
        
        if method not in self.analyzer.clusters_per_method:
            messagebox.showwarning("Warnung", "Keine Cluster-Daten! Erst Analyse durchführen.")
            return
        
        self.clear_viz()
        
        fig = Figure(figsize=(8, 6), facecolor=COLORS['bg_plot'])
        ax = fig.add_subplot(111, facecolor=COLORS['bg_plot'])
        
        clusters = self.analyzer.clusters_per_method[method]
        method_info = AGGREGATION_METHODS[method]
        
        cluster_names = list(clusters.keys())
        n = len(cluster_names)
        
        hamming_matrix = np.zeros((n, n))
        for i, c1 in enumerate(cluster_names):
            for j, c2 in enumerate(cluster_names):
                p1 = clusters[c1]['pattern']
                p2 = clusters[c2]['pattern']
                hamming_matrix[i, j] = sum(a != b for a, b in zip(p1, p2))
        
        im = ax.imshow(hamming_matrix, cmap='Blues')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(cluster_names)
        ax.set_yticklabels(cluster_names)
        ax.set_title(f'Hamming-Distanzen zwischen Clustern ({method_info["short"]})', 
                     fontsize=14, fontweight='bold')
        
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{int(hamming_matrix[i, j])}',
                        ha='center', va='center', fontsize=12)
        
        fig.colorbar(im, ax=ax, label='Hamming-Distanz')
        fig.tight_layout()
        
        self.current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_figure(self):
        """Speichert aktuelle Figur"""
        if self.current_fig is None:
            messagebox.showwarning("Warnung", "Keine Visualisierung vorhanden!")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filepath:
            self.current_fig.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Erfolg", f"Gespeichert: {filepath}")
    
    def run(self):
        """Startet GUI"""
        self.root.mainloop()


def run_cli(files, order=None, output_dir=None, aggregation_method='v0'):
    """CLI-Modus für Analyse ohne GUI"""
    analyzer = LayerAnalyzer()
    
    for filepath in files:
        name = Path(filepath).stem
        print(f"Lade: {name}")
        analyzer.load_layer(filepath, name)
    
    if order is None:
        order = sorted(analyzer.layers.keys(), key=natural_sort_key)
    analyzer.set_layer_order(order)
    
    print("\nBerechne Korrelationen (alle Methoden)...")
    analyzer.compute_all_correlations()
    
    print("Identifiziere Cluster (alle Methoden)...")
    analyzer.identify_all_clusters()
    
    print("Extrahiere Bitstrings (8 Methoden)...")
    analyzer.extract_combined_bitstrings()
    
    summary = analyzer.get_summary(aggregation_method)
    print(summary)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Hauptanalyse
        json_path = output_path / "gcis_analysis.json"
        analyzer.export_json(str(json_path), aggregation_method)
        print(f"\nJSON exportiert: {json_path}")
        
        txt_path = output_path / "gcis_analysis.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Text exportiert: {txt_path}")
        
        # Einzelne Bitstring-Methoden
        for method, info in BITSTRING_METHODS.items():
            method_txt = output_path / f"bitstring_{method}_{info['short']}.txt"
            with open(method_txt, 'w', encoding='utf-8') as f:
                f.write(analyzer.get_method_summary(method))
            
            method_json = output_path / f"bitstring_{method}_{info['short']}.json"
            analyzer.export_method_json(method, str(method_json))
        
        print(f"\n8 Bitstring-Methoden-Dateien (TXT + JSON) exportiert.")
    
    return analyzer


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='GCIS Layer Analyzer v6.0 - Korrelation pro Methode')
    parser.add_argument('--gui', action='store_true', help='Starte GUI-Modus')
    parser.add_argument('--files', nargs='*', help='CSV-Dateien zur Analyse')
    parser.add_argument('--folder', help='Ordner mit CSV-Dateien')
    parser.add_argument('--order', nargs='*', help='Layer-Reihenfolge')
    parser.add_argument('--output', help='Ausgabe-Verzeichnis')
    parser.add_argument('--aggregation', default='v0', choices=['v0', 'v2', 'v3'],
                        help='Aggregationsmethode für Korrelation (v0=Summe, v2=Skewness, v3=Median)')
    
    args = parser.parse_args()
    
    if args.gui or (not args.files and not args.folder):
        if HAS_TKINTER:
            app = AnalyzerGUI()
            app.run()
        else:
            print("GUI-Modus nicht verfügbar (tkinter fehlt).")
            print("Verwende CLI-Modus: python script.py --files *.csv")
    else:
        files = []
        if args.files:
            files.extend(args.files)
        if args.folder:
            files.extend(sorted(Path(args.folder).glob("*.csv")))
        
        if not files:
            print("Keine Dateien gefunden!")
            sys.exit(1)
        
        run_cli(files, order=args.order, output_dir=args.output, 
                aggregation_method=args.aggregation)