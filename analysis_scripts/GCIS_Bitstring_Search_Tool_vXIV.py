#!/usr/bin/env python3
"""
===============================================================================
GCIS BITSTRING SEARCH TOOL vXIV - PREPRINT EDITION
===============================================================================
Features:
- Selective layer export (â‰¥80% threshold)
- Publication-ready plots (Heatmap, Bar Chart, Scatter)
- Separate Password/Hash analysis modes
- Raw bitstring export for reproducibility
- LaTeX table generation
===============================================================================
"""

import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from collections import defaultdict
import csv
import os
from datetime import datetime

# Plot imports
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not found. Plots disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("WARNING: numpy not found. Some features disabled.")

# PDF export with reportlab
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not found. PDF export disabled.")

# UI Colors
COLORS = {
    'gold': '#D4A574',
    'gold_dark': '#B8935F',
    'gold_light': '#E6C8A0',
    'dark': '#1A1A1A',
    'light_gray': '#F8F8F8',
    'white': '#FFFFFF',
    'green': '#28A745',
    'blue': '#007BFF',
    'red': '#DC3545',
    'purple': '#6F42C1',
}

# Scientific color palette (Nature/Science/Cell style)
SCIENCE_COLORS = {
    'primary': '#0077B6',      # Deep blue - main findings
    'secondary': '#00B4D8',    # Cyan - secondary data
    'accent': '#FF6B35',       # Orange - highlights/found positions
    'neutral': '#6B7280',      # Gray - background/context
    'success': '#059669',      # Emerald - 100% match
    'warning': '#D97706',      # Amber - partial match
    'background': '#F8FAFC',   # Light gray background
    'grid': '#E2E8F0',         # Grid lines
    'text': '#1E293B',         # Dark text
    'bitstring_0': '#E2E8F0',  # Light gray for '0' bits
    'bitstring_1': '#94A3B8',  # Medium gray for '1' bits
    'found_marker': '#DC2626', # Red for found positions
}

# Publication-ready plot style
PLOT_STYLE = {
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#E2E8F0',
    'axes.facecolor': '#FAFBFC',
    'figure.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 0.8,
}


class SearchResultsExportDialog(tk.Toplevel):
    """Dialog for exporting search results to CSV, TXT, and PDF"""
    
    def __init__(self, parent, results, search_bytes, layer_sizes, data_type, bit_length,
                 identifier, search_string, layers_data):
        super().__init__(parent)
        self.title("Export Search Results")
        self.geometry("500x350")
        self.configure(bg=COLORS['white'])
        
        self.results = results
        self.search_bytes = search_bytes
        self.layer_sizes = layer_sizes
        self.data_type = data_type
        self.bit_length = bit_length
        self.identifier = identifier
        self.search_string = search_string
        self.layers_data = layers_data
        self.parent_gui = parent
        
        self.setup_ui()
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS['gold_light'])
        header.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(header, text="Export Search Results",
                 font=('Verdana', 14, 'bold'), bg=COLORS['gold_light']).pack(pady=5)
        tk.Label(header, text="Exports CSV + TXT + PDF simultaneously",
                 font=('Verdana', 9), bg=COLORS['gold_light']).pack()
        
        # Filter Frame
        filter_frame = tk.LabelFrame(self, text="Filter by Match Rate",
                                      bg=COLORS['light_gray'], font=('Verdana', 10, 'bold'))
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        filter_row = tk.Frame(filter_frame, bg=COLORS['light_gray'])
        filter_row.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(filter_row, text="Minimum Match Rate:", bg=COLORS['light_gray'],
                 font=('Verdana', 10)).pack(side=tk.LEFT, padx=5)
        
        self.filter_var = tk.StringVar(value="100%")
        filter_options = ["100%", "â‰¥90%", "â‰¥60%", "â‰¥40%", "All"]
        self.filter_combo = ttk.Combobox(filter_row, textvariable=self.filter_var,
                                          values=filter_options, state="readonly", width=10)
        self.filter_combo.pack(side=tk.LEFT, padx=10)
        self.filter_combo.bind('<<ComboboxSelected>>', self.update_preview)
        
        # Preview Label
        self.preview_label = tk.Label(filter_frame, text="", bg=COLORS['light_gray'],
                                       font=('Verdana', 10, 'bold'), fg=COLORS['gold_dark'])
        self.preview_label.pack(pady=5)
        
        # Info Frame
        info_frame = tk.LabelFrame(self, text="Export Information",
                                    bg=COLORS['light_gray'], font=('Verdana', 10, 'bold'))
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = tk.Frame(info_frame, bg=COLORS['light_gray'])
        info_text.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(info_text, text=f"Data Type: {self.data_type}", bg=COLORS['light_gray'],
                 font=('Courier', 9)).pack(anchor='w')
        tk.Label(info_text, text=f"Search: {len(self.search_bytes)} x {self.bit_length}-bit strings",
                 bg=COLORS['light_gray'], font=('Courier', 9)).pack(anchor='w')
        tk.Label(info_text, text=f"Identifier: {self.identifier or '(not set)'}",
                 bg=COLORS['light_gray'], font=('Courier', 9)).pack(anchor='w')
        
        # Button Frame
        btn_frame = tk.Frame(self, bg=COLORS['white'])
        btn_frame.pack(fill=tk.X, padx=10, pady=20)
        
        tk.Button(btn_frame, text="ðŸ“ Select Folder & Export",
                  command=self.do_export,
                  bg=COLORS['green'], fg=COLORS['white'],
                  font=('Verdana', 11, 'bold')).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Cancel",
                  command=self.destroy,
                  bg=COLORS['gold_light'],
                  font=('Verdana', 10)).pack(side=tk.LEFT, padx=10)
        
        # Initial preview
        self.update_preview()
    
    def get_threshold(self):
        """Returns the threshold as a fraction based on selected filter"""
        filter_val = self.filter_var.get()
        if filter_val == "100%":
            return 1.0
        elif filter_val == "â‰¥94%":
            return 0.94
        elif filter_val == "â‰¥60%":
            return 0.6
        elif filter_val == "â‰¥40%":
            return 0.4
        else:  # All
            return 0.0
    
    def get_filtered_layers(self):
        """Returns layers that meet the threshold"""
        threshold = self.get_threshold()
        total_bytes = len(self.search_bytes)
        min_matches = int(total_bytes * threshold)
        
        filtered = []
        for layer_name, found_bytes in self.results.items():
            hits = len(found_bytes)
            if hits >= min_matches:
                filtered.append((layer_name, found_bytes, hits))
        
        # Sort by hits descending, then by name
        filtered.sort(key=lambda x: (-x[2], x[0]))
        return filtered
    
    def update_preview(self, event=None):
        """Updates the preview label"""
        filtered = self.get_filtered_layers()
        total_bytes = len(self.search_bytes)
        
        full_match = sum(1 for _, _, hits in filtered if hits == total_bytes)
        n_minus_1 = sum(1 for _, _, hits in filtered if hits == total_bytes - 1)
        
        self.preview_label.config(
            text=f"Will export: {len(filtered)} layers "
                 f"(100%: {full_match}, N-1: {n_minus_1})"
        )
    
    def do_export(self):
        """Performs the export to all three formats"""
        filtered = self.get_filtered_layers()
        
        if not filtered:
            messagebox.showwarning("Warning", "No layers match the selected filter!")
            return
        
        # Select folder
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        
        # Generate base filename
        safe_id = re.sub(r'[^\w\-]', '_', self.identifier) if self.identifier else "export"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"search_results_{safe_id}_{timestamp}"
        
        exported_files = []
        
        try:
            # Export CSV
            csv_path = os.path.join(folder, f"{base_name}.csv")
            self._export_csv(csv_path, filtered)
            exported_files.append(csv_path)
            
            # Export TXT
            txt_path = os.path.join(folder, f"{base_name}.txt")
            self._export_txt(txt_path, filtered)
            exported_files.append(txt_path)
            
            # Export PDF
            if REPORTLAB_AVAILABLE:
                pdf_path = os.path.join(folder, f"{base_name}.pdf")
                self._export_pdf(pdf_path, filtered)
                exported_files.append(pdf_path)
            else:
                messagebox.showwarning("Warning", "reportlab not installed - PDF export skipped!")
            
            # Success message
            msg = f"Exported {len(exported_files)} files:\n\n"
            msg += "\n".join([os.path.basename(f) for f in exported_files])
            msg += f"\n\nFolder: {folder}"
            messagebox.showinfo("Export Complete", msg)
            
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _format_positions(self, positions, max_show=5):
        """Format positions list with limit"""
        if len(positions) <= max_show:
            return ', '.join(str(p) for p in positions)
        else:
            shown = ', '.join(str(p) for p in positions[:max_show])
            return f"{shown}... (+{len(positions) - max_show})"
    
    def _export_csv(self, filepath, filtered):
        """Export to CSV format"""
        total_bytes = len(self.search_bytes)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['GCIS Search Results Export'])
            writer.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Data Type', self.data_type])
            writer.writerow(['Identifier', self.identifier])
            writer.writerow(['Search', f'{total_bytes} x {self.bit_length}-bit strings'])
            writer.writerow(['Filter', self.filter_var.get()])
            writer.writerow([])
            
            # Search String
            writer.writerow(['SEARCH STRING'])
            writer.writerow([self.search_string])
            writer.writerow([])
            
            # Statistics Summary
            writer.writerow(['STATISTICS SUMMARY'])
            writer.writerow(['Layer', 'Bits', 'Matches', 'Match_Rate_%', 'Coverage_%'])
            
            for layer_name, found_bytes, hits in filtered:
                size = self.layer_sizes.get(layer_name, 0)
                pct = (hits / total_bytes) * 100
                coverage = (hits * self.bit_length / size * 100) if size > 0 else 0
                writer.writerow([layer_name, size, f'{hits}/{total_bytes}', f'{pct:.1f}', f'{coverage:.4f}'])
            
            writer.writerow([])
            
            # Layer Details
            writer.writerow(['LAYER DETAILS'])
            writer.writerow(['Layer', 'Bitstring', 'Char', 'Positions'])
            
            for layer_name, found_bytes, hits in filtered:
                for byte_str, positions in found_bytes.items():
                    char = self._byte_to_char(byte_str)
                    pos_str = self._format_positions(positions)
                    writer.writerow([layer_name, byte_str, char, pos_str])
    
    def _export_txt(self, filepath, filtered):
        """Export to TXT format"""
        total_bytes = len(self.search_bytes)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("GCIS SEARCH RESULTS EXPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Type: {self.data_type}\n")
            f.write(f"Identifier: {self.identifier}\n")
            f.write(f"Search: {total_bytes} x {self.bit_length}-bit strings\n")
            f.write(f"Filter: {self.filter_var.get()}\n")
            f.write("\n")
            
            # Search String
            f.write("-" * 80 + "\n")
            f.write("SEARCH STRING\n")
            f.write("-" * 80 + "\n")
            f.write(f"{self.search_string}\n")
            f.write("\n")
            
            # Distribution
            f.write("-" * 80 + "\n")
            f.write("DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            
            count_distribution = defaultdict(list)
            for layer_name, found_bytes, hits in filtered:
                count_distribution[hits].append(layer_name)
            
            for count in sorted(count_distribution.keys(), reverse=True):
                layer_list = count_distribution[count]
                percentage = (count / total_bytes) * 100
                f.write(f"{count:2d}/{total_bytes} ({percentage:5.1f}%): {len(layer_list):3d} layers\n")
            
            f.write("\n")
            
            # 100% Match Layers
            full_match = [(n, fb, h) for n, fb, h in filtered if h == total_bytes]
            f.write("-" * 80 + "\n")
            f.write(f"100% MATCH LAYERS ({total_bytes}/{total_bytes}):\n")
            f.write("-" * 80 + "\n")
            
            if full_match:
                for layer_name, found_bytes, hits in full_match:
                    size = self.layer_sizes.get(layer_name, 0)
                    coverage = (hits * self.bit_length / size * 100) if size > 0 else 0
                    f.write(f"  {layer_name} ({size} bits, coverage = {coverage:.4f}%)\n")
            else:
                f.write("  (none)\n")
            
            f.write("\n")
            
            # N-1 Match Layers
            n_minus_1 = [(n, fb, h) for n, fb, h in filtered if h == total_bytes - 1]
            f.write(f"N-1 MATCH LAYERS ({total_bytes-1}/{total_bytes}):\n")
            f.write("-" * 80 + "\n")
            
            if n_minus_1:
                for layer_name, found_bytes, hits in n_minus_1:
                    size = self.layer_sizes.get(layer_name, 0)
                    coverage = (hits * self.bit_length / size * 100) if size > 0 else 0
                    f.write(f"  {layer_name} ({size} bits, coverage = {coverage:.4f}%)\n")
            else:
                f.write("  (none)\n")
            
            f.write("\n")
            
            # Layer Details
            f.write("=" * 80 + "\n")
            f.write("LAYER DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            for layer_name, found_bytes, hits in filtered:
                size = self.layer_sizes.get(layer_name, 0)
                pct = (hits / total_bytes) * 100
                
                if hits == total_bytes:
                    f.write(f"* {layer_name}: {hits}/{total_bytes} (100%) - {size} bits\n")
                elif hits == total_bytes - 1:
                    f.write(f"o {layer_name}: {hits}/{total_bytes} ({pct:.0f}%) - {size} bits\n")
                else:
                    f.write(f"  {layer_name}: {hits}/{total_bytes} ({pct:.0f}%) - {size} bits\n")
                
                for byte_str, positions in found_bytes.items():
                    char = self._byte_to_char(byte_str)
                    pos_str = self._format_positions(positions)
                    f.write(f"    {byte_str} '{char}' @ {pos_str}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
    
    def _export_pdf(self, filepath, filtered):
        """Export to PDF format with tables"""
        total_bytes = len(self.search_bytes)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                                rightMargin=1.5*cm, leftMargin=1.5*cm,
                                topMargin=1.5*cm, bottomMargin=1.5*cm)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                      fontSize=16, spaceAfter=12)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                        fontSize=12, spaceAfter=6, spaceBefore=12)
        normal_style = styles['Normal']
        
        elements = []
        
        # Title
        elements.append(Paragraph("GCIS Search Results Export", title_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # Metadata Table
        meta_data = [
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Data Type', self.data_type],
            ['Identifier', self.identifier or '(not set)'],
            ['Search', f'{total_bytes} x {self.bit_length}-bit strings'],
            ['Filter', self.filter_var.get()],
            ['Layers Exported', str(len(filtered))]
        ]
        
        meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), rl_colors.Color(0.83, 0.65, 0.45)),  # Gold
            ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 0.5*cm))
        
        # Statistics Summary Table
        elements.append(Paragraph("Statistics Summary", heading_style))
        
        stats_header = ['Layer', 'Bits', 'Matches', 'Rate %', 'Coverage %']
        stats_data = [stats_header]
        
        for layer_name, found_bytes, hits in filtered:
            size = self.layer_sizes.get(layer_name, 0)
            pct = (hits / total_bytes) * 100
            coverage = (hits * self.bit_length / size * 100) if size > 0 else 0
            stats_data.append([layer_name, str(size), f'{hits}/{total_bytes}',
                              f'{pct:.1f}', f'{coverage:.4f}'])
        
        stats_table = Table(stats_data, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 2*cm, 3*cm])
        
        # Build table style with conditional colors
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.Color(0.83, 0.65, 0.45)),  # Header gold
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]
        
        # Color rows based on match rate
        for i, (layer_name, found_bytes, hits) in enumerate(filtered, start=1):
            if hits == total_bytes:
                table_style.append(('BACKGROUND', (0, i), (-1, i), rl_colors.Color(0.78, 0.95, 0.78)))  # Light green
            elif hits == total_bytes - 1:
                table_style.append(('BACKGROUND', (0, i), (-1, i), rl_colors.Color(0.78, 0.85, 0.95)))  # Light blue
        
        stats_table.setStyle(TableStyle(table_style))
        elements.append(stats_table)
        
        # Page break before Layer Details
        elements.append(PageBreak())
        
        # Layer Details
        elements.append(Paragraph("Layer Details", title_style))
        elements.append(Spacer(1, 0.3*cm))
        
        for layer_name, found_bytes, hits in filtered:
            size = self.layer_sizes.get(layer_name, 0)
            pct = (hits / total_bytes) * 100
            
            # Layer header
            if hits == total_bytes:
                status = "â˜… 100%"
                header_color = rl_colors.Color(0.16, 0.65, 0.27)  # Green
            elif hits == total_bytes - 1:
                status = "â—‹ N-1"
                header_color = rl_colors.Color(0, 0.48, 1)  # Blue
            else:
                status = f"  {pct:.0f}%"
                header_color = rl_colors.grey
            
            layer_header = Paragraph(
                f"<b>{layer_name}</b>: {hits}/{total_bytes} ({status}) - {size} bits",
                ParagraphStyle('LayerHeader', fontSize=10, textColor=header_color)
            )
            elements.append(layer_header)
            elements.append(Spacer(1, 0.2*cm))
            
            # Positions table for this layer
            pos_header = ['Bitstring', 'Char', 'Positions']
            pos_data = [pos_header]
            
            for byte_str, positions in found_bytes.items():
                char = self._byte_to_char(byte_str)
                pos_str = self._format_positions(positions)
                pos_data.append([byte_str, char, pos_str])
            
            pos_table = Table(pos_data, colWidths=[3*cm, 1.5*cm, 12*cm])
            pos_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.Color(0.9, 0.9, 0.9)),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.3, rl_colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(pos_table)
            elements.append(Spacer(1, 0.4*cm))
        
        doc.build(elements)
    
    def _byte_to_char(self, byte_str):
        """Convert binary string to character representation"""
        try:
            bit_len = len(byte_str)
            val = int(byte_str, 2)
            
            if bit_len == 8:
                if 32 <= val <= 126:
                    return chr(val)
                else:
                    return '.'
            elif bit_len % 8 == 0 and 8 < bit_len <= 32:
                hex_digits = bit_len // 4
                hex_str = format(val, f'0{hex_digits}X')
                num_chars = bit_len // 8
                ascii_chars = []
                for i in range(num_chars):
                    byte_start = i * 8
                    byte_end = byte_start + 8
                    byte_bits = byte_str[byte_start:byte_end]
                    byte_val = int(byte_bits, 2)
                    if 32 <= byte_val <= 126:
                        ascii_chars.append(chr(byte_val))
                    else:
                        ascii_chars.append('.')
                ascii_str = ''.join(ascii_chars)
                return f"{hex_str} ({ascii_str})"
            elif bit_len == 4:
                return format(val, 'X')
            elif bit_len == 12:
                return format(val, '03X')
            else:
                hex_digits = (bit_len + 3) // 4
                return format(val, f'0{hex_digits}X')
        except:
            return '?'


class ExportDialog(tk.Toplevel):
    """Dialog for selective layer export with checkboxes"""
    
    def __init__(self, parent, layers_data, search_bytes, results, layer_sizes, 
                 data_type="Password", bit_length=8):
        super().__init__(parent)
        self.title(f"Export Selection - {data_type}")
        self.geometry("900x700")
        self.configure(bg=COLORS['white'])
        
        self.layers_data = layers_data
        self.search_bytes = search_bytes
        self.results = results
        self.layer_sizes = layer_sizes
        self.data_type = data_type
        self.bit_length = bit_length
        self.selected_layers = []
        self.checkboxes = {}
        
        self.setup_ui()
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS['gold_light'])
        header.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(header, text=f"Select Layers for Export ({self.data_type})",
                 font=('Verdana', 12, 'bold'), bg=COLORS['gold_light']).pack(pady=5)
        
        tk.Label(header, text="Only layers with â‰¥94% match rate are available",
                 font=('Verdana', 9), bg=COLORS['gold_light']).pack()
        
        # Filter info
        total_bytes = len(self.search_bytes)
        threshold = int(total_bytes * 0.8)
        
        # Layer selection frame with scrollbar
        select_frame = tk.LabelFrame(self, text="Available Layers (â‰¥94%)",
                                      bg=COLORS['light_gray'], font=('Verdana', 10, 'bold'))
        select_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas for scrolling
        canvas = tk.Canvas(select_frame, bg=COLORS['light_gray'])
        scrollbar = ttk.Scrollbar(select_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['light_gray'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header row
        header_frame = tk.Frame(scrollable_frame, bg=COLORS['gold_light'])
        header_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(header_frame, text="Select", width=8, bg=COLORS['gold_light'],
                 font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="Layer", width=15, bg=COLORS['gold_light'],
                 font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="Match", width=12, bg=COLORS['gold_light'],
                 font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="Layer Bits", width=12, bg=COLORS['gold_light'],
                 font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="String Coverage", width=15, bg=COLORS['gold_light'],
                 font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Populate layers
        eligible_layers = []
        for layer_name, found_bytes in self.results.items():
            hits = len(found_bytes)
            if hits >= threshold:
                eligible_layers.append((layer_name, hits, found_bytes))
        
        # Sort by hits descending
        eligible_layers.sort(key=lambda x: (-x[1], x[0]))
        
        for layer_name, hits, found_bytes in eligible_layers:
            percentage = (hits / total_bytes) * 100
            size = self.layer_sizes.get(layer_name, 0)
            coverage = (hits * self.bit_length / size * 100) if size > 0 else 0
            
            row_frame = tk.Frame(scrollable_frame, bg=COLORS['white'])
            row_frame.pack(fill=tk.X, padx=5, pady=1)
            
            var = tk.BooleanVar(value=(hits == total_bytes))  # Pre-select 100% matches
            cb = tk.Checkbutton(row_frame, variable=var, bg=COLORS['white'])
            cb.pack(side=tk.LEFT, padx=5)
            self.checkboxes[layer_name] = var
            
            # Color based on match rate
            if hits == total_bytes:
                color = COLORS['green']
            elif hits == total_bytes - 1:
                color = COLORS['blue']
            else:
                color = COLORS['dark']
            
            tk.Label(row_frame, text=layer_name, width=15, bg=COLORS['white'],
                     fg=color, font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
            tk.Label(row_frame, text=f"{hits}/{total_bytes} ({percentage:.1f}%)", 
                     width=12, bg=COLORS['white'], fg=color,
                     font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
            tk.Label(row_frame, text=f"{size:,}", width=12, bg=COLORS['white'],
                     font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
            tk.Label(row_frame, text=f"{coverage:.4f}%", width=15, bg=COLORS['white'],
                     font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Selection buttons
        btn_frame = tk.Frame(self, bg=COLORS['white'])
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="Select All", command=self.select_all,
                  bg=COLORS['gold_light']).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select None", command=self.select_none,
                  bg=COLORS['gold_light']).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select 100% Only", command=self.select_full,
                  bg=COLORS['green'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        # Export options
        export_frame = tk.LabelFrame(self, text="Export Options",
                                      bg=COLORS['light_gray'], font=('Verdana', 10, 'bold'))
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.export_plots = tk.BooleanVar(value=True)
        self.export_csv = tk.BooleanVar(value=True)
        self.export_latex = tk.BooleanVar(value=True)
        self.export_bitstrings = tk.BooleanVar(value=True)
        
        opts_row = tk.Frame(export_frame, bg=COLORS['light_gray'])
        opts_row.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(opts_row, text="Plots (PNG)", variable=self.export_plots,
                       bg=COLORS['light_gray']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(opts_row, text="Summary CSV", variable=self.export_csv,
                       bg=COLORS['light_gray']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(opts_row, text="LaTeX Table", variable=self.export_latex,
                       bg=COLORS['light_gray']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(opts_row, text="Raw Bitstrings", variable=self.export_bitstrings,
                       bg=COLORS['light_gray']).pack(side=tk.LEFT, padx=10)
        
        # Action buttons
        action_frame = tk.Frame(self, bg=COLORS['white'])
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(action_frame, text="EXPORT SELECTED",
                  command=self.do_export,
                  bg=COLORS['green'], fg=COLORS['white'],
                  font=('Verdana', 11, 'bold')).pack(side=tk.LEFT, padx=10)
        
        tk.Button(action_frame, text="Cancel",
                  command=self.destroy,
                  bg=COLORS['gold_light'],
                  font=('Verdana', 10)).pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status_label = tk.Label(action_frame, text="", bg=COLORS['white'])
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.update_status()
    
    def select_all(self):
        for var in self.checkboxes.values():
            var.set(True)
        self.update_status()
    
    def select_none(self):
        for var in self.checkboxes.values():
            var.set(False)
        self.update_status()
    
    def select_full(self):
        total_bytes = len(self.search_bytes)
        for layer_name, var in self.checkboxes.items():
            hits = len(self.results.get(layer_name, {}))
            var.set(hits == total_bytes)
        self.update_status()
    
    def update_status(self):
        count = sum(1 for var in self.checkboxes.values() if var.get())
        self.status_label.config(text=f"{count} layers selected")
    
    def do_export(self):
        self.selected_layers = [name for name, var in self.checkboxes.items() if var.get()]
        
        if not self.selected_layers:
            messagebox.showwarning("Warning", "No layers selected!")
            return
        
        self.export_options = {
            'plots': self.export_plots.get(),
            'csv': self.export_csv.get(),
            'latex': self.export_latex.get(),
            'bitstrings': self.export_bitstrings.get(),
        }
        
        self.destroy()


class PlotWindow(tk.Toplevel):
    """Window for displaying and saving plots"""
    
    def __init__(self, parent, plot_type, data, title, data_type="Password"):
        super().__init__(parent)
        self.title(f"{title} - {data_type}")
        self.geometry("1000x700")
        self.configure(bg=COLORS['white'])
        
        self.plot_type = plot_type
        self.data = data
        self.plot_title = title
        self.data_type = data_type
        self.figure = None
        
        self.setup_ui()
        self.create_plot()
    
    def setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self, bg=COLORS['gold_light'])
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(toolbar, text="Save PNG (300 DPI)",
                  command=lambda: self.save_plot('png'),
                  bg=COLORS['green'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        tk.Button(toolbar, text="Save SVG",
                  command=lambda: self.save_plot('svg'),
                  bg=COLORS['blue'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        tk.Button(toolbar, text="Save PDF",
                  command=lambda: self.save_plot('pdf'),
                  bg=COLORS['purple'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        # Plot canvas
        self.canvas_frame = tk.Frame(self, bg=COLORS['white'])
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_plot(self):
        if not MATPLOTLIB_AVAILABLE:
            tk.Label(self.canvas_frame, text="matplotlib not available",
                     font=('Verdana', 14)).pack(expand=True)
            return
        
        # Apply scientific style
        for key, value in PLOT_STYLE.items():
            try:
                plt.rcParams[key] = value
            except:
                pass
        
        self.figure = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        
        if self.plot_type == 'heatmap':
            self.create_heatmap()
        elif self.plot_type == 'bar':
            self.create_bar_chart()
        elif self.plot_type == 'bitstring':
            self.create_bitstring_plot()
        
        canvas = FigureCanvasTkAgg(self.figure, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_heatmap(self):
        """Heatmap of bit positions per layer - scientific style"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(SCIENCE_COLORS['background'])
        
        layers = self.data['layers']
        search_bytes = self.data['search_bytes']
        results = self.data['results']
        layer_sizes = self.data['layer_sizes']
        
        # Build matrix
        n_layers = len(layers)
        n_bytes = len(search_bytes)
        
        if n_layers == 0 or n_bytes == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   color=SCIENCE_COLORS['text'])
            return
        
        matrix = []
        y_labels = []
        
        for layer_name in layers:
            found = results.get(layer_name, {})
            row = []
            for byte in search_bytes:
                if byte in found:
                    # Normalize position to 0-1 range
                    pos = found[byte][0]
                    size = layer_sizes.get(layer_name, 1)
                    normalized = pos / size
                    row.append(normalized)
                else:
                    row.append(-0.1)  # Missing = distinct color
            matrix.append(row)
            y_labels.append(layer_name)
        
        if NUMPY_AVAILABLE:
            matrix = np.array(matrix)
        
        # Create custom colormap (scientific blue-orange)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#DC2626', '#0077B6', '#00B4D8', '#38BDF8', '#7DD3FC']
        cmap = LinearSegmentedColormap.from_list('scientific', colors_list)
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-0.1, vmax=1.0)
        
        # Labels with scientific styling
        ax.set_xlabel('Byte Index in Search String', fontsize=10, 
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_ylabel('Layer', fontsize=10, 
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_title(f'Preimage Bit Position Localization ({self.data_type})\nNormalized Position within Layer Output', 
                     fontsize=11, fontweight='bold', color=SCIENCE_COLORS['text'], pad=15)
        
        # X ticks
        if n_bytes <= 30:
            ax.set_xticks(range(n_bytes))
            ax.set_xticklabels([str(i+1) for i in range(n_bytes)], fontsize=8)
        
        # Y ticks
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(y_labels, fontsize=9, fontfamily='monospace')
        
        # Colorbar with scientific styling
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Normalized Position (0=start, 1=end, red=missing)', 
                      fontsize=9, color=SCIENCE_COLORS['text'])
        cbar.ax.tick_params(labelsize=8)
        
        self.figure.tight_layout()
    
    def create_bar_chart(self):
        """Bar chart: Match percentage vs layer size - scientific style"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(SCIENCE_COLORS['background'])
        
        layers = self.data['layers']
        search_bytes = self.data['search_bytes']
        results = self.data['results']
        layer_sizes = self.data['layer_sizes']
        
        total_bytes = len(search_bytes)
        
        # Prepare data
        layer_names = []
        sizes = []
        percentages = []
        colors = []
        
        for layer_name in sorted(layers, key=lambda x: layer_sizes.get(x, 0)):
            found = results.get(layer_name, {})
            hits = len(found)
            pct = (hits / total_bytes) * 100 if total_bytes > 0 else 0
            size = layer_sizes.get(layer_name, 0)
            
            layer_names.append(layer_name)
            sizes.append(size)
            percentages.append(pct)
            
            if hits == total_bytes:
                colors.append(SCIENCE_COLORS['success'])  # 100% match
            elif hits == total_bytes - 1:
                colors.append(SCIENCE_COLORS['primary'])  # N-1 match
            else:
                colors.append(SCIENCE_COLORS['warning'])  # <N-1 match
        
        x = range(len(layer_names))
        
        # Create bars with scientific styling
        bars = ax.bar(x, percentages, color=colors, edgecolor=SCIENCE_COLORS['text'], 
                     linewidth=0.5, alpha=0.85)
        
        # Add size labels on bars
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            ax.annotate(f'{size:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=45,
                        color=SCIENCE_COLORS['text'])
        
        ax.set_xlabel('Layer (sorted by size)', fontsize=10, 
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_ylabel('Match Rate (%)', fontsize=10,
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_title(f'Preimage Match Rate vs. Layer Size ({self.data_type})\nLabels indicate layer bit count', 
                     fontsize=11, fontweight='bold', color=SCIENCE_COLORS['text'], pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8,
                          fontfamily='monospace')
        ax.set_ylim(0, 105)
        
        # Grid styling
        ax.yaxis.grid(True, color=SCIENCE_COLORS['grid'], linestyle='-', linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        
        # Legend with scientific colors
        legend_elements = [
            mpatches.Patch(facecolor=SCIENCE_COLORS['success'], 
                          edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                          label='100% Match'),
            mpatches.Patch(facecolor=SCIENCE_COLORS['primary'],
                          edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                          label='N-1 Match'),
            mpatches.Patch(facecolor=SCIENCE_COLORS['warning'],
                          edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                          label='<N-1 Match'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
                 edgecolor=SCIENCE_COLORS['grid'])
        
        # 80% threshold line
        ax.axhline(y=80, color=SCIENCE_COLORS['found_marker'], linestyle='--', 
                  linewidth=1.2, alpha=0.7, label='80% Threshold')
        
        self.figure.tight_layout()
    
    def create_bitstring_plot(self):
        """Visualize bitstring with found positions highlighted"""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(SCIENCE_COLORS['background'])
        
        layer_name = self.data.get('current_layer', '')
        bitstring = self.data.get('bitstring', '')
        found_positions = self.data.get('found_positions', {})
        search_bytes = self.data.get('search_bytes', [])
        bit_length = self.data.get('bit_length', 8)
        
        if not bitstring:
            ax.text(0.5, 0.5, 'No bitstring data available', ha='center', va='center',
                   color=SCIENCE_COLORS['text'], fontsize=12)
            return
        
        # Limit display for very long bitstrings
        max_display = 2048
        display_bitstring = bitstring[:max_display] if len(bitstring) > max_display else bitstring
        truncated = len(bitstring) > max_display
        
        n_bits = len(display_bitstring)
        
        # Calculate grid dimensions
        bits_per_row = min(128, n_bits)  # 128 bits per row for readability
        n_rows = (n_bits + bits_per_row - 1) // bits_per_row
        
        if NUMPY_AVAILABLE:
            # Create image matrix
            img_data = np.zeros((n_rows, bits_per_row, 3))
            
            # Fill with bitstring values
            for i, bit in enumerate(display_bitstring):
                row = i // bits_per_row
                col = i % bits_per_row
                
                if bit == '0':
                    img_data[row, col] = [0.886, 0.910, 0.941]  # Light gray #E2E8F0
                else:
                    img_data[row, col] = [0.580, 0.639, 0.718]  # Medium gray #94A3B8
            
            # Mark found positions with red
            for byte_str, positions in found_positions.items():
                for pos in positions:
                    if pos < max_display:
                        # Mark the entire byte length
                        for offset in range(bit_length):
                            bit_pos = pos + offset
                            if bit_pos < n_bits:
                                row = bit_pos // bits_per_row
                                col = bit_pos % bits_per_row
                                img_data[row, col] = [0.863, 0.149, 0.149]  # Red #DC2626
            
            ax.imshow(img_data, aspect='auto', interpolation='nearest')
        else:
            ax.text(0.5, 0.5, 'numpy required for bitstring visualization', 
                   ha='center', va='center', color=SCIENCE_COLORS['text'])
            return
        
        # Styling
        ax.set_xlabel(f'Bit Position (column, {bits_per_row} bits/row)', fontsize=10,
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_ylabel('Row', fontsize=10, color=SCIENCE_COLORS['text'], fontweight='medium')
        
        title = f'Preimage Localization in Layer: {layer_name} ({self.data_type})'
        if truncated:
            title += f'\nShowing first {max_display:,} of {len(bitstring):,} bits'
        ax.set_title(title, fontsize=11, fontweight='bold', 
                    color=SCIENCE_COLORS['text'], pad=15)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#E2E8F0', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label="Bit '0'"),
            mpatches.Patch(facecolor='#94A3B8', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label="Bit '1'"),
            mpatches.Patch(facecolor='#DC2626', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label='Found Preimage Byte'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
                 edgecolor=SCIENCE_COLORS['grid'], fontsize=8)
        
        # Statistics annotation
        total_found = sum(len(positions) for positions in found_positions.values())
        n_bytes_found = len(found_positions)
        stats_text = f'Found: {n_bytes_found}/{len(search_bytes)} bytes ({total_found} occurrences)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', color=SCIENCE_COLORS['text'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                        edgecolor=SCIENCE_COLORS['grid']))
        
        self.figure.tight_layout()
    
    def save_plot(self, fmt):
        if not self.figure:
            return
        
        filetypes = {
            'png': [("PNG files", "*.png")],
            'svg': [("SVG files", "*.svg")],
            'pdf': [("PDF files", "*.pdf")],
        }
        
        filepath = filedialog.asksaveasfilename(
            title=f"Save Plot as {fmt.upper()}",
            defaultextension=f".{fmt}",
            filetypes=filetypes.get(fmt, [("All files", "*.*")])
        )
        
        if filepath:
            dpi = 300 if fmt == 'png' else 150
            self.figure.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            messagebox.showinfo("Success", f"Plot saved to:\n{filepath}")


class BitstringSearchGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GCIS Bitstring Search Tool vXIII - Preprint Edition")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLORS['white'])
        
        self.layers = {}
        self.layer_sizes = {}
        self.filepath = None
        self.last_results = None
        self.last_search_bytes = None
        self.last_bit_length = 8
        self.selected_export_layers = []  # Stores the last selected export layers
        
        # Data type tracking
        self.current_data_type = tk.StringVar(value="Password")
        
        self.setup_gui()
    
    def setup_gui(self):
        # === TOP: File loading ===
        top_frame = tk.Frame(self.root, bg=COLORS['white'])
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(top_frame, text="Load TXT File",
                  command=self.load_file,
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(top_frame, text="No file loaded",
                                    bg=COLORS['white'], fg=COLORS['dark'])
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        self.layer_count_label = tk.Label(top_frame, text="",
                                           bg=COLORS['white'], fg=COLORS['gold_dark'],
                                           font=('Verdana', 10, 'bold'))
        self.layer_count_label.pack(side=tk.LEFT, padx=10)
        
        # Data type selector
        tk.Label(top_frame, text="Data Type:", bg=COLORS['white'],
                 font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=(30, 5))
        
        type_menu = ttk.Combobox(top_frame, textvariable=self.current_data_type,
                                  values=["Password", "Hash (MD5)", "Hash (SHA-256)", 
                                         "Hash (SHA-512)", "Custom"],
                                  state="readonly", width=15)
        type_menu.pack(side=tk.LEFT, padx=5)
        
        # === INPUT: Password/Identifier ===
        pw_frame = tk.LabelFrame(self.root, text="IDENTIFIER (for filename)",
                                  bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                  font=('Verdana', 10, 'bold'))
        pw_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.password_entry = tk.Entry(pw_frame, font=('Courier', 11))
        self.password_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # === INPUT: Comment ===
        comment_frame = tk.LabelFrame(self.root, text="COMMENT (optional - for export)",
                                  bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                  font=('Verdana', 10, 'bold'))
        comment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.comment_entry = tk.Entry(comment_frame, font=('Courier', 11))
        self.comment_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # === INPUT: Binary string ===
        input_frame = tk.LabelFrame(self.root, text="SEARCH STRING (Binary, space-separated, 1-32 bit auto-detect)",
                                     bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                     font=('Verdana', 10, 'bold'))
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label for detected bit length
        bit_frame = tk.Frame(input_frame, bg=COLORS['light_gray'])
        bit_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(bit_frame, text="Detected Bit Length:", bg=COLORS['light_gray']).pack(side=tk.LEFT, padx=5)
        self.detected_bit_label = tk.Label(bit_frame, text="(not detected yet)", 
                                           bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                           font=('Courier', 10, 'bold'))
        self.detected_bit_label.pack(side=tk.LEFT, padx=5)
        
        self.search_entry = tk.Text(input_frame, height=3, font=('Courier', 11))
        self.search_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # Button row 1: Search
        btn_frame1 = tk.Frame(input_frame, bg=COLORS['light_gray'])
        btn_frame1.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame1, text="SEARCH",
                  command=self.search,
                  bg=COLORS['gold'], fg=COLORS['dark'],
                  font=('Verdana', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame1, text="Clear",
                  command=lambda: self.search_entry.delete('1.0', tk.END),
                  bg=COLORS['gold_light'], fg=COLORS['dark'],
                  font=('Verdana', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame1, text="Missing String Analysis",
                  command=self.analyze_missing,
                  bg=COLORS['red'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=20)
        
        # Button row 2: Export
        btn_frame2 = tk.Frame(input_frame, bg=COLORS['light_gray'])
        btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame2, text="ðŸ“Š PREPRINT EXPORT",
                  command=self.open_export_dialog,
                  bg=COLORS['purple'], fg=COLORS['white'],
                  font=('Verdana', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame2, text="ðŸ“ˆ Heatmap",
                  command=lambda: self.show_plot('heatmap'),
                  bg=COLORS['blue'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame2, text="ðŸ“Š Bar Chart",
                  command=lambda: self.show_plot('bar'),
                  bg=COLORS['blue'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame2, text="ðŸ”¬ Bitstring Map",
                  command=self.show_bitstring_selector,
                  bg=COLORS['green'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame2, text="ðŸ“ Position Correlation",
                  command=self.show_position_correlation,
                  bg=COLORS['purple'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Legacy export buttons
        tk.Button(btn_frame2, text="CSV (Simple)",
                  command=self.export_csv_full,
                  bg=COLORS['green'], fg=COLORS['white'],
                  font=('Verdana', 9)).pack(side=tk.LEFT, padx=20)
        
        # NEW: Full Export (CSV+TXT+PDF)
        tk.Button(btn_frame2, text="ðŸ“„ Export Results",
                  command=self.open_search_export_dialog,
                  bg=COLORS['blue'], fg=COLORS['white'],
                  font=('Verdana', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # === RESULTS: 3 columns ===
        results_frame = tk.Frame(self.root, bg=COLORS['white'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Statistics
        left_frame = tk.LabelFrame(results_frame, text="STATISTICS",
                                    bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                    font=('Verdana', 10, 'bold'))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.stats_text = scrolledtext.ScrolledText(left_frame, font=('Courier', 10),
                                                      wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_text.tag_configure('green', foreground=COLORS['green'], font=('Courier', 10, 'bold'))
        self.stats_text.tag_configure('blue', foreground=COLORS['blue'], font=('Courier', 10, 'bold'))
        self.stats_text.tag_configure('red', foreground=COLORS['red'], font=('Courier', 10, 'bold'))
        self.stats_text.tag_configure('gold', foreground=COLORS['gold_dark'], font=('Courier', 10, 'bold'))
        
        # Middle: Details
        mid_frame = tk.LabelFrame(results_frame, text="LAYER DETAILS",
                                   bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                   font=('Verdana', 10, 'bold'))
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.details_text = scrolledtext.ScrolledText(mid_frame, font=('Courier', 10),
                                                        wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.details_text.tag_configure('green', foreground=COLORS['green'], font=('Courier', 10, 'bold'))
        self.details_text.tag_configure('blue', foreground=COLORS['blue'], font=('Courier', 10, 'bold'))
        
        # Right: Missing analysis
        right_frame = tk.LabelFrame(results_frame, text="MISSING STRING & POSITIONS",
                                     bg=COLORS['light_gray'], fg=COLORS['gold_dark'],
                                     font=('Verdana', 10, 'bold'))
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.missing_text = scrolledtext.ScrolledText(right_frame, font=('Courier', 10),
                                                        wrap=tk.WORD)
        self.missing_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.missing_text.tag_configure('red', foreground=COLORS['red'], font=('Courier', 10, 'bold'))
        self.missing_text.tag_configure('green', foreground=COLORS['green'], font=('Courier', 10, 'bold'))
        self.missing_text.tag_configure('gold', foreground=COLORS['gold_dark'], font=('Courier', 10, 'bold'))
    
    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Load Analysis TXT",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        self.filepath = filepath
        self.layers = {}
        self.layer_sizes = {}
        
        try:
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except:
                    continue
            
            if content is None:
                raise Exception("Could not read file")
            
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            pattern = r'([a-zA-Z0-9_]+):\s*(\d+)\s*Neurons,\s*Bitstring:\s*([01]+)'
            matches = re.findall(pattern, content)
            
            for name, neurons, bitstring in matches:
                self.layers[name] = bitstring
                self.layer_sizes[name] = len(bitstring)
            
            self.file_label.config(text=f"File: {os.path.basename(filepath)}")
            self.layer_count_label.config(text=f"{len(self.layers)} layers loaded")
            
            if len(self.layers) == 0:
                messagebox.showwarning("Warning", "No layers found!")
            else:
                messagebox.showinfo("Info", f"{len(self.layers)} layers loaded.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file:\n{e}")
    
    def search(self):
        if not self.layers:
            messagebox.showwarning("Warning", "Load a file first!")
            return
        
        search_text = self.search_entry.get('1.0', tk.END).strip()
        if not search_text:
            messagebox.showwarning("Warning", "No search string entered!")
            return
        
        bytes_to_find = search_text.split()
        
        # AUTO-DETECT bit length from first valid string
        detected_length = None
        valid_bytes = []
        
        for b in bytes_to_find:
            b = b.strip()
            if not b:
                continue
            
            # Check if it's a valid binary string
            if not all(c in '01' for c in b):
                messagebox.showerror("Error", f"Invalid binary string: '{b}'\nOnly 0 and 1 allowed!")
                return
            
            length = len(b)
            
            # Check length limits
            if length < 1 or length > 32:
                messagebox.showerror("Error", f"String '{b}' has {length} bits.\nAllowed: 1-32 bits!")
                return
            
            # Detect length from first string
            if detected_length is None:
                detected_length = length
            
            # Validate all strings have same length
            if length != detected_length:
                messagebox.showerror("Error", 
                    f"Inconsistent bit lengths!\n"
                    f"First string: {detected_length} bits\n"
                    f"Found string with: {length} bits ('{b}')\n\n"
                    f"All strings must have the same length!")
                return
            
            valid_bytes.append(b)
        
        if not valid_bytes:
            messagebox.showwarning("Warning", "No valid binary strings found!")
            return
        
        # Update detected bit length display
        self.detected_bit_label.config(
            text=f"{detected_length} bits ({len(valid_bytes)} strings)",
            fg=COLORS['green']
        )
        
        # Store detected length
        bit_len = detected_length
        self.last_bit_length = bit_len
        
        results = {}
        
        for layer_name, bitstring in self.layers.items():
            results[layer_name] = {}
            for byte in valid_bytes:
                positions = []
                start = 0
                while True:
                    pos = bitstring.find(byte, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                if positions:
                    results[layer_name][byte] = positions
        
        self.last_results = results
        self.last_search_bytes = valid_bytes
        
        self.display_results(valid_bytes, results)
    
    def display_results(self, search_bytes, results):
        total_bytes = len(search_bytes)
        bit_len = self.last_bit_length
        data_type = self.current_data_type.get()
        
        self.stats_text.delete('1.0', tk.END)
        
        count_distribution = defaultdict(list)
        
        for layer_name, found_bytes in results.items():
            count = len(found_bytes)
            count_distribution[count].append(layer_name)
        
        self.stats_text.insert(tk.END, f"DATA TYPE: {data_type}\n", 'gold')
        self.stats_text.insert(tk.END, f"SEARCH: {total_bytes} x {bit_len}-bit strings\n")
        self.stats_text.insert(tk.END, "=" * 50 + "\n\n")
        
        self.stats_text.insert(tk.END, "DISTRIBUTION:\n")
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        for count in sorted(count_distribution.keys(), reverse=True):
            layer_list = count_distribution[count]
            percentage = (count / total_bytes) * 100 if total_bytes > 0 else 0
            
            line = f"{count:2d}/{total_bytes} ({percentage:5.1f}%): {len(layer_list):3d} layers\n"
            
            if count == total_bytes:
                self.stats_text.insert(tk.END, line, 'green')
            elif count == total_bytes - 1:
                self.stats_text.insert(tk.END, line, 'blue')
            else:
                self.stats_text.insert(tk.END, line)
        
        full_match = count_distribution.get(total_bytes, [])
        self.stats_text.insert(tk.END, "\n")
        self.stats_text.insert(tk.END, "=" * 50 + "\n")
        self.stats_text.insert(tk.END, f"\n100% MATCH LAYERS ({total_bytes}/{total_bytes}):\n", 'green')
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        if full_match:
            for name in sorted(full_match):
                size = self.layer_sizes.get(name, 0)
                byte_percentage = (total_bytes * bit_len / size * 100) if size > 0 else 0
                self.stats_text.insert(tk.END, f"  {name} ({size} bits, coverage = {byte_percentage:.4f}%)\n", 'green')
        else:
            self.stats_text.insert(tk.END, "  (none)\n")
        
        minus_one = count_distribution.get(total_bytes - 1, [])
        self.stats_text.insert(tk.END, f"\nN-1 MATCH LAYERS ({total_bytes-1}/{total_bytes}):\n", 'blue')
        self.stats_text.insert(tk.END, "-" * 50 + "\n")
        
        if minus_one:
            for name in sorted(minus_one):
                size = self.layer_sizes.get(name, 0)
                byte_percentage = ((total_bytes-1) * bit_len / size * 100) if size > 0 else 0
                self.stats_text.insert(tk.END, f"  {name} ({size} bits, coverage = {byte_percentage:.4f}%)\n", 'blue')
        else:
            self.stats_text.insert(tk.END, "  (none)\n")
        
        # Layer details
        self.details_text.delete('1.0', tk.END)
        
        sorted_layers = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
        
        for layer_name, found_bytes in sorted_layers:
            count = len(found_bytes)
            if count == 0:
                continue
            
            percentage = (count / total_bytes) * 100
            size = self.layer_sizes.get(layer_name, 0)
            
            if count == total_bytes:
                self.details_text.insert(tk.END, f"* {layer_name}: {count}/{total_bytes} (100%) - {size} bits\n", 'green')
            elif count == total_bytes - 1:
                self.details_text.insert(tk.END, f"o {layer_name}: {count}/{total_bytes} ({percentage:.0f}%) - {size} bits\n", 'blue')
            else:
                self.details_text.insert(tk.END, f"  {layer_name}: {count}/{total_bytes} ({percentage:.0f}%) - {size} bits\n")
            
            for byte, positions in found_bytes.items():
                char = self.byte_to_char(byte)
                pos_str = ', '.join(str(p) for p in positions[:5])
                if len(positions) > 5:
                    pos_str += f"... (+{len(positions)-5})"
                self.details_text.insert(tk.END, f"    {byte} '{char}' @ {pos_str}\n")
            
            self.details_text.insert(tk.END, "\n")
    
    def show_bitstring_selector(self):
        """Show dialog to select layer for bitstring visualization"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error", "matplotlib is not installed!")
            return
        
        total_bytes = len(self.last_search_bytes)
        threshold = int(total_bytes * 0.8)
        
        # Filter layers >= 80%
        eligible_layers = [(name, len(found)) for name, found in self.last_results.items() 
                          if len(found) >= threshold]
        
        if not eligible_layers:
            messagebox.showwarning("Warning", "No layers with â‰¥80% match rate!")
            return
        
        # Sort by hits descending
        eligible_layers.sort(key=lambda x: (-x[1], x[0]))
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Layer for Bitstring Visualization")
        dialog.geometry("400x300")
        dialog.configure(bg=COLORS['white'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select a layer to visualize:", 
                bg=COLORS['white'], font=('Verdana', 10, 'bold')).pack(pady=10)
        
        # Listbox with layers
        listbox_frame = tk.Frame(dialog, bg=COLORS['white'])
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, font=('Courier', 10), 
                            yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for name, hits in eligible_layers:
            pct = (hits / total_bytes) * 100
            size = self.layer_sizes.get(name, 0)
            listbox.insert(tk.END, f"{name} - {hits}/{total_bytes} ({pct:.0f}%) - {size:,} bits")
        
        listbox.selection_set(0)  # Select first by default
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                layer_name = eligible_layers[selection[0]][0]
                dialog.destroy()
                self.show_bitstring_plot(layer_name)
        
        btn_frame = tk.Frame(dialog, bg=COLORS['white'])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btn_frame, text="Show Visualization", command=on_select,
                  bg=COLORS['green'], fg=COLORS['white'],
                  font=('Verdana', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                  bg=COLORS['gold_light']).pack(side=tk.LEFT, padx=5)
    
    def show_bitstring_plot(self, layer_name):
        """Show bitstring visualization for a specific layer"""
        bitstring = self.layers.get(layer_name, '')
        found = self.last_results.get(layer_name, {})
        
        data = {
            'current_layer': layer_name,
            'bitstring': bitstring,
            'found_positions': found,
            'search_bytes': self.last_search_bytes,
            'bit_length': self.last_bit_length,
            'layers': [layer_name],
            'results': self.last_results,
            'layer_sizes': self.layer_sizes,
        }
        
        PlotWindow(self.root, 'bitstring', data, 
                   f'Bitstring Localization: {layer_name}',
                   self.current_data_type.get())
    
    def show_position_correlation(self):
        """Run position correlation on previously selected export layers"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            messagebox.showerror("Error", "matplotlib and numpy required!")
            return
        
        # Use the last selected export layers
        if not self.selected_export_layers or len(self.selected_export_layers) < 2:
            messagebox.showwarning("Warning", 
                "First use PREPRINT EXPORT to select layers (min 2).\n"
                "Position Correlation uses that selection.")
            return
        
        selected_layers = self.selected_export_layers
        
        # Build position matrices
        abs_positions = {}
        norm_positions = {}
        
        for layer_name in selected_layers:
            found = self.last_results.get(layer_name, {})
            size = self.layer_sizes.get(layer_name, 1)
            
            abs_positions[layer_name] = {}
            norm_positions[layer_name] = {}
            
            for byte_str, positions in found.items():
                pos = positions[0]
                abs_positions[layer_name][byte_str] = pos
                norm_positions[layer_name][byte_str] = pos / size
        
        # Analyze correlations
        analysis_results = self.analyze_position_patterns(
            selected_layers, abs_positions, norm_positions, self.last_search_bytes, self.layer_sizes
        )
        
        # Create visualization
        self.show_position_correlation_plot(
            selected_layers, abs_positions, norm_positions, 
            self.last_search_bytes, analysis_results
        )
    
    def analyze_position_patterns(self, layer_names, abs_positions, norm_positions, search_bytes, layer_sizes):
        """Analyze if bytes appear at same absolute or relative positions across layers
        
        TOLERANCE BANDS:
        - es-Layer (â‰¥1024 bits): Â±16 bits bandwidth (abs_range â‰¤ 32)
        - zfa-Layer (<1024 bits): Â±8 bits bandwidth (abs_range â‰¤ 16)
        """
        results = {
            'absolute_matches': [],  # Bytes with overlapping tolerance bands
            'relative_matches': [],  # Bytes that appear at same relative position (Â±0.01)
            'byte_stats': {},  # Per-byte statistics
            'layer_sizes': layer_sizes,  # Store for display
        }
        
        for i, byte_str in enumerate(search_bytes):
            byte_stats = {
                'byte_index': i + 1,
                'byte_value': byte_str,
                'char': self.byte_to_char(byte_str),
                'abs_positions': [],
                'norm_positions': [],
                'abs_std': None,
                'norm_std': None,
                'abs_match': False,
                'norm_match': False,
            }
            
            # Collect positions across layers WITH layer info for tolerance calculation
            layer_positions = {}  # {layer_name: position}
            for layer_name in layer_names:
                if byte_str in abs_positions[layer_name]:
                    layer_positions[layer_name] = abs_positions[layer_name][byte_str]
                    byte_stats['abs_positions'].append(abs_positions[layer_name][byte_str])
                    byte_stats['norm_positions'].append(norm_positions[layer_name][byte_str])
            
            if len(byte_stats['abs_positions']) >= 2:
                # Calculate standard deviation
                abs_arr = np.array(byte_stats['abs_positions'])
                norm_arr = np.array(byte_stats['norm_positions'])
                
                byte_stats['abs_std'] = np.std(abs_arr)
                byte_stats['norm_std'] = np.std(norm_arr)
                byte_stats['abs_range'] = np.max(abs_arr) - np.min(abs_arr)
                byte_stats['norm_range'] = np.max(norm_arr) - np.min(norm_arr)
                
                # Check if tolerance bands overlap
                # es-Layer (â‰¥1024 bits): Â±16 bits â†’ band = [pos-16, pos+16]
                # zfa-Layer (<1024 bits): Â±8 bits â†’ band = [pos-8, pos+8]
                bands = []
                for layer_name, pos in layer_positions.items():
                    size = layer_sizes.get(layer_name, 0)
                    if size >= 1024:
                        tolerance = 16  # es-Layer: Â±16 bits
                    else:
                        tolerance = 8   # zfa-Layer: Â±8 bits
                    bands.append((pos - tolerance, pos + tolerance))
                
                # Check if all bands overlap (intersection is non-empty)
                if bands:
                    intersection_low = max(b[0] for b in bands)
                    intersection_high = min(b[1] for b in bands)
                    if intersection_low <= intersection_high:
                        byte_stats['abs_match'] = True
                        results['absolute_matches'].append(i + 1)
                
                if byte_stats['norm_range'] <= 0.01:  # 1% of layer size
                    byte_stats['norm_match'] = True
                    results['relative_matches'].append(i + 1)
            
            results['byte_stats'][i + 1] = byte_stats
        
        return results
    
    def show_position_correlation_plot(self, layer_names, abs_positions, norm_positions, 
                                        search_bytes, analysis):
        """Show position correlation visualization in TWO separate windows"""
        
        for key, value in PLOT_STYLE.items():
            try:
                plt.rcParams[key] = value
            except:
                pass
        
        data_type = self.current_data_type.get()
        n_bytes = len(search_bytes)
        x = np.arange(n_bytes)
        layer_colors = plt.cm.Set2(np.linspace(0, 1, len(layer_names)))
        
        # =====================================================================
        # WINDOW 1: Position Plots (Absolute + Normalized)
        # =====================================================================
        fig1 = plt.figure(figsize=(14, 5), facecolor='white')
        
        ax1 = fig1.add_subplot(121)  # Absolute positions
        ax2 = fig1.add_subplot(122)  # Normalized positions
        
        # Plot 1: Absolute positions
        ax1.set_facecolor(SCIENCE_COLORS['background'])
        for idx, layer_name in enumerate(layer_names):
            positions = []
            for byte_str in search_bytes:
                if byte_str in abs_positions[layer_name]:
                    positions.append(abs_positions[layer_name][byte_str])
                else:
                    positions.append(np.nan)
            ax1.plot(x, positions, 'o-', label=layer_name, color=layer_colors[idx], 
                    markersize=4, linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Byte Index', fontsize=10, color=SCIENCE_COLORS['text'])
        ax1.set_ylabel('Absolute Position (bits)', fontsize=10, color=SCIENCE_COLORS['text'])
        ax1.set_title('Absolute Bit Positions Across Layers', fontsize=11, 
                     fontweight='bold', color=SCIENCE_COLORS['text'])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i+1) for i in x], fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized positions
        ax2.set_facecolor(SCIENCE_COLORS['background'])
        for idx, layer_name in enumerate(layer_names):
            positions = []
            for byte_str in search_bytes:
                if byte_str in norm_positions[layer_name]:
                    positions.append(norm_positions[layer_name][byte_str])
                else:
                    positions.append(np.nan)
            ax2.plot(x, positions, 'o-', label=layer_name, color=layer_colors[idx],
                    markersize=4, linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Byte Index', fontsize=10, color=SCIENCE_COLORS['text'])
        ax2.set_ylabel('Normalized Position (0-1)', fontsize=10, color=SCIENCE_COLORS['text'])
        ax2.set_title('Normalized Positions Across Layers\n(Position / Layer Size)', 
                     fontsize=11, fontweight='bold', color=SCIENCE_COLORS['text'])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(i+1) for i in x], fontsize=7)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        fig1.suptitle(f'Cross-Layer Position Correlation Analysis ({data_type})',
                    fontsize=13, fontweight='bold', color=SCIENCE_COLORS['text'], y=0.98)
        fig1.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Window 1
        plot_window1 = tk.Toplevel(self.root)
        plot_window1.title(f"Position Correlation - Positions ({data_type})")
        plot_window1.geometry("1200x500")
        
        toolbar1 = tk.Frame(plot_window1, bg=COLORS['gold_light'])
        toolbar1.pack(fill=tk.X, padx=5, pady=5)
        
        def save_plot1(fmt):
            filepath = filedialog.asksaveasfilename(
                title=f"Save Positions Plot as {fmt.upper()}",
                defaultextension=f".{fmt}",
                filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")]
            )
            if filepath:
                dpi = 300 if fmt == 'png' else 150
                fig1.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Saved to:\n{filepath}")
        
        tk.Button(toolbar1, text="Save PNG", command=lambda: save_plot1('png'),
                  bg=COLORS['green'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar1, text="Save SVG", command=lambda: save_plot1('svg'),
                  bg=COLORS['blue'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        canvas1 = FigureCanvasTkAgg(fig1, plot_window1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # WINDOW 2: Match Score + Summary (SEPARATE WINDOW)
        # =====================================================================
        fig2 = plt.figure(figsize=(14, 6), facecolor='#FFFFFF')
        
        ax3 = fig2.add_subplot(121)  # Match Score bar chart
        ax4 = fig2.add_subplot(122)  # Summary text
        
        # Calculate MATCH SCORES (inverted: 100% = perfect match)
        match_scores = []
        bar_colors = []
        
        for i in range(n_bytes):
            stats = analysis['byte_stats'].get(i + 1, {})
            norm_range = stats.get('norm_range', 1.0) or 1.0  # Default to 1.0 (no match)
            
            # Match Score: 100% wenn Range=0, 0% wenn Range>=1 (100%)
            # Score = 100 * (1 - norm_range), clamped to [0, 100]
            score = max(0, min(100, 100 * (1 - norm_range)))
            match_scores.append(score)
            
            # Color based on score - elegant scientific color scheme
            if score >= 98:
                bar_colors.append('#2E5A88')  # Dark blue (Excellent)
            elif score >= 90:
                bar_colors.append('#4374B3')  # Medium blue (Highly Significant)
            elif score >= 75:
                bar_colors.append('#7BA3D0')  # Light blue (Remarkable)
            else:
                bar_colors.append('#D4A574')  # Gold (Below threshold)
        
        # Bar chart with elegant styling
        bars = ax3.bar(x, match_scores, color=bar_colors, alpha=0.9, edgecolor='#666666', linewidth=0.3)
        
        # Threshold lines - elegant scientific style
        ax3.axhline(y=98, color='#2E5A88', linestyle='-', 
                   linewidth=2, label='98% (Excellent)', alpha=0.8)
        ax3.axhline(y=90, color='#4374B3', linestyle='--', 
                   linewidth=1.5, label='90% (Highly Significant)')
        ax3.axhline(y=75, color='#7BA3D0', linestyle=':', 
                   linewidth=1.5, label='75% (Remarkable)')
        
        ax3.set_xlabel('Byte Index', fontsize=11, color='#1A1A1A')
        ax3.set_ylabel('Position Match Score (%)', fontsize=11, color='#1A1A1A')
        ax3.set_title('Position Consistency Match Score\n(Higher = Better Match)', 
                     fontsize=12, fontweight='bold', color='#1A1A1A', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(i+1) for i in x], fontsize=8)
        ax3.set_ylim(0, 105)
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y', color='#B8935F')
        ax3.set_facecolor('#FDF6EC')
        
        # Add value labels on bars for excellent matches (â‰¥98%)
        for i, (bar, score) in enumerate(zip(bars, match_scores)):
            if score >= 98:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold',
                        color='#2E5A88')
        
        # Plot 4: Summary statistics
        ax4.set_facecolor('#FDF6EC')
        ax4.axis('off')
        
        # Categorize bytes by match score ranges
        range_98_100 = [i+1 for i, s in enumerate(match_scores) if s >= 98]
        range_90_97 = [i+1 for i, s in enumerate(match_scores) if 90 <= s < 98]
        range_75_89 = [i+1 for i, s in enumerate(match_scores) if 75 <= s < 90]
        
        # Determine bit length from the first search byte
        bit_len = len(search_bytes[0]) if search_bytes else 8
        
        summary_text = f"""POSITION CORRELATION ANALYSIS
{'='*50}

Data Type: {data_type}
Layers Analyzed: {len(layer_names)}
Bytestring size {bit_len}bit Analysed: {n_bytes}

TOLERANCE BANDS:
  es-Layer (â‰¥1024 bits): Â±16 bits bandwidth
  zfa-Layer (<1024 bits): Â±8 bits bandwidth

MATCH SCORE DISTRIBUTION:
{'='*50}

98-100% (Excellent):
  {len(range_98_100)}/{n_bytes} bytes ({100*len(range_98_100)/n_bytes:.1f}%)
  Indices: {range_98_100 if range_98_100 else 'None'}

90-97% (Highly Significant):
  {len(range_90_97)}/{n_bytes} bytes ({100*len(range_90_97)/n_bytes:.1f}%)
  Indices: {range_90_97 if range_90_97 else 'None'}

75-89% (Remarkable):
  {len(range_75_89)}/{n_bytes} bytes ({100*len(range_75_89)/n_bytes:.1f}%)
  Indices: {range_75_89 if range_75_89 else 'None'}

TOTAL â‰¥75%: {len(range_98_100) + len(range_90_97) + len(range_75_89)}/{n_bytes} bytes
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                fontfamily='monospace', verticalalignment='top',
                color='#1A1A1A',
                bbox=dict(boxstyle='round', facecolor='#FFFFFF', alpha=0.95,
                         edgecolor='#B8935F', linewidth=1.5))
        
        fig2.suptitle(f'Position Match Analysis ({data_type})',
                    fontsize=13, fontweight='bold', color='#1A1A1A', y=0.98)
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Window 2
        plot_window2 = tk.Toplevel(self.root)
        plot_window2.title(f"Position Match Score + Summary ({data_type})")
        plot_window2.geometry("1200x550")
        
        toolbar2 = tk.Frame(plot_window2, bg=COLORS['gold_light'])
        toolbar2.pack(fill=tk.X, padx=5, pady=5)
        
        def save_plot2(fmt):
            filepath = filedialog.asksaveasfilename(
                title=f"Save Match Score Plot as {fmt.upper()}",
                defaultextension=f".{fmt}",
                filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")]
            )
            if filepath:
                dpi = 300 if fmt == 'png' else 150
                fig2.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Saved to:\n{filepath}")
        
        tk.Button(toolbar2, text="Save PNG", command=lambda: save_plot2('png'),
                  bg=COLORS['green'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar2, text="Save SVG", command=lambda: save_plot2('svg'),
                  bg=COLORS['blue'], fg=COLORS['white']).pack(side=tk.LEFT, padx=5)
        
        canvas2 = FigureCanvasTkAgg(fig2, plot_window2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_plot(self, plot_type):
        """Open a plot window"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error", "matplotlib is not installed!\nInstall with: pip install matplotlib")
            return
        
        total_bytes = len(self.last_search_bytes)
        threshold = int(total_bytes * 0.8)
        
        # Filter layers >= 80%
        eligible_layers = [name for name, found in self.last_results.items() 
                          if len(found) >= threshold]
        
        if not eligible_layers:
            messagebox.showwarning("Warning", "No layers with â‰¥80% match rate!")
            return
        
        data = {
            'layers': eligible_layers,
            'search_bytes': self.last_search_bytes,
            'results': self.last_results,
            'layer_sizes': self.layer_sizes,
            'bit_length': self.last_bit_length,
        }
        
        titles = {
            'heatmap': 'Bit Position Localization Heatmap',
            'bar': 'Match Rate vs. Layer Size',
            'scatter': 'String Coverage vs. Match Rate',
        }
        
        PlotWindow(self.root, plot_type, data, titles[plot_type], 
                   self.current_data_type.get())
    
    def open_export_dialog(self):
        """Open the preprint export dialog"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        dialog = ExportDialog(
            self.root,
            self.layers,
            self.last_search_bytes,
            self.last_results,
            self.layer_sizes,
            self.current_data_type.get(),
            self.last_bit_length
        )
        
        self.root.wait_window(dialog)
        
        if hasattr(dialog, 'selected_layers') and dialog.selected_layers:
            self.selected_export_layers = dialog.selected_layers  # Store for Position Correlation
            self.do_preprint_export(dialog.selected_layers, dialog.export_options)
    
    def do_preprint_export(self, selected_layers, options):
        """Execute the preprint export"""
        if not self.filepath:
            messagebox.showwarning("Warning", "No source directory!")
            return
        
        identifier = self.password_entry.get().strip() or "export"
        safe_id = re.sub(r'[^\w\-]', '_', identifier)
        data_type = self.current_data_type.get().replace(" ", "_").replace("(", "").replace(")", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        source_dir = os.path.dirname(self.filepath)
        export_dir = os.path.join(source_dir, f"preprint_export_{safe_id}_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = []
        
        # 1. Export plots
        if options.get('plots') and MATPLOTLIB_AVAILABLE:
            # Heatmap and Bar Chart (one each)
            for plot_type in ['heatmap', 'bar']:
                fig = self.create_plot_figure(plot_type, selected_layers)
                if fig:
                    for fmt in ['png', 'svg']:
                        filepath = os.path.join(export_dir, f"plot_{plot_type}_{data_type}.{fmt}")
                        dpi = 300 if fmt == 'png' else 150
                        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
                        exported_files.append(filepath)
                    plt.close(fig)
            
            # Bitstring plots - one per layer
            for layer_name in selected_layers:
                fig = self.create_bitstring_figure(layer_name)
                if fig:
                    safe_layer = re.sub(r'[^\w\-]', '_', layer_name)
                    for fmt in ['png', 'svg']:
                        filepath = os.path.join(export_dir, f"bitstring_{safe_layer}_{data_type}.{fmt}")
                        dpi = 300 if fmt == 'png' else 150
                        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
                        exported_files.append(filepath)
                    plt.close(fig)
        
        # 2. Export summary CSV
        if options.get('csv'):
            filepath = os.path.join(export_dir, f"summary_{data_type}.csv")
            self.export_summary_csv(filepath, selected_layers)
            exported_files.append(filepath)
        
        # 3. Export LaTeX table
        if options.get('latex'):
            filepath = os.path.join(export_dir, f"table_{data_type}.tex")
            self.export_latex_table(filepath, selected_layers)
            exported_files.append(filepath)
        
        # 4. Export raw bitstrings
        if options.get('bitstrings'):
            filepath = os.path.join(export_dir, f"bitstrings_{data_type}.csv")
            self.export_raw_bitstrings(filepath, selected_layers)
            exported_files.append(filepath)
        
        # Success message
        msg = f"Exported {len(exported_files)} files to:\n{export_dir}\n\nFiles:\n"
        msg += "\n".join([os.path.basename(f) for f in exported_files[:15]])
        if len(exported_files) > 15:
            msg += f"\n... and {len(exported_files) - 15} more"
        messagebox.showinfo("Export Complete", msg)
    
    def create_bitstring_figure(self, layer_name):
        """Create a bitstring visualization figure for a single layer"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        
        for key, value in PLOT_STYLE.items():
            try:
                plt.rcParams[key] = value
            except:
                pass
        
        fig = Figure(figsize=(14, 8), dpi=150, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor(SCIENCE_COLORS['background'])
        
        bitstring = self.layers.get(layer_name, '')
        found = self.last_results.get(layer_name, {})
        data_type = self.current_data_type.get()
        bit_length = self.last_bit_length
        
        if not bitstring:
            return None
        
        # Limit display for very long bitstrings
        max_display = 4096
        display_bitstring = bitstring[:max_display] if len(bitstring) > max_display else bitstring
        truncated = len(bitstring) > max_display
        
        n_bits = len(display_bitstring)
        
        # Calculate grid dimensions
        bits_per_row = min(256, n_bits)  # 256 bits per row for export
        n_rows = (n_bits + bits_per_row - 1) // bits_per_row
        
        # Create image matrix
        img_data = np.zeros((n_rows, bits_per_row, 3))
        
        # Fill with bitstring values
        for i, bit in enumerate(display_bitstring):
            row = i // bits_per_row
            col = i % bits_per_row
            
            if bit == '0':
                img_data[row, col] = [0.886, 0.910, 0.941]  # Light gray
            else:
                img_data[row, col] = [0.580, 0.639, 0.718]  # Medium gray
        
        # Mark found positions with red
        for byte_str, positions in found.items():
            for pos in positions:
                if pos < max_display:
                    for offset in range(bit_length):
                        bit_pos = pos + offset
                        if bit_pos < n_bits:
                            row = bit_pos // bits_per_row
                            col = bit_pos % bits_per_row
                            img_data[row, col] = [0.863, 0.149, 0.149]  # Red
        
        ax.imshow(img_data, aspect='auto', interpolation='nearest')
        
        # Styling
        ax.set_xlabel(f'Bit Position (column, {bits_per_row} bits/row)', fontsize=10,
                     color=SCIENCE_COLORS['text'], fontweight='medium')
        ax.set_ylabel('Row', fontsize=10, color=SCIENCE_COLORS['text'], fontweight='medium')
        
        title = f'Preimage Localization in Layer: {layer_name} ({data_type})'
        subtitle = f'Layer size: {len(bitstring):,} bits'
        if truncated:
            subtitle += f' | Showing first {max_display:,} bits'
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold', 
                    color=SCIENCE_COLORS['text'], pad=15)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#E2E8F0', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label="Bit '0'"),
            mpatches.Patch(facecolor='#94A3B8', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label="Bit '1'"),
            mpatches.Patch(facecolor='#DC2626', edgecolor=SCIENCE_COLORS['text'],
                          linewidth=0.5, label='Found Preimage Byte'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
                 edgecolor=SCIENCE_COLORS['grid'], fontsize=9)
        
        # Statistics
        total_found = sum(len(positions) for positions in found.values())
        n_bytes_found = len(found)
        total_search = len(self.last_search_bytes)
        stats_text = f'Match: {n_bytes_found}/{total_search} bytes ({(n_bytes_found/total_search*100):.1f}%) | {total_found} occurrences'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', color=SCIENCE_COLORS['text'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor=SCIENCE_COLORS['grid']))
        
        fig.tight_layout()
        return fig
    
    def create_plot_figure(self, plot_type, selected_layers):
        """Create a matplotlib figure for export"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        for key, value in PLOT_STYLE.items():
            try:
                plt.rcParams[key] = value
            except:
                pass
        
        fig = Figure(figsize=(12, 7), dpi=150, facecolor='white')
        data_type = self.current_data_type.get()
        
        total_bytes = len(self.last_search_bytes)
        
        if plot_type == 'heatmap':
            ax = fig.add_subplot(111)
            ax.set_facecolor(SCIENCE_COLORS['background'])
            
            matrix = []
            y_labels = []
            
            for layer_name in selected_layers:
                found = self.last_results.get(layer_name, {})
                row = []
                for byte in self.last_search_bytes:
                    if byte in found:
                        pos = found[byte][0]
                        size = self.layer_sizes.get(layer_name, 1)
                        normalized = pos / size
                        row.append(normalized)
                    else:
                        row.append(-0.1)
                matrix.append(row)
                y_labels.append(layer_name)
            
            if NUMPY_AVAILABLE:
                matrix = np.array(matrix)
            
            # Scientific colormap
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#DC2626', '#0077B6', '#00B4D8', '#38BDF8', '#7DD3FC']
            cmap = LinearSegmentedColormap.from_list('scientific', colors_list)
            
            im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-0.1, vmax=1.0)
            
            ax.set_xlabel('Byte Index in Search String', fontsize=10,
                         color=SCIENCE_COLORS['text'], fontweight='medium')
            ax.set_ylabel('Layer', fontsize=10,
                         color=SCIENCE_COLORS['text'], fontweight='medium')
            ax.set_title(f'Preimage Bit Position Localization ({data_type})\nNormalized Position within Layer Output', 
                         fontsize=11, fontweight='bold', color=SCIENCE_COLORS['text'], pad=15)
            
            n_bytes = len(self.last_search_bytes)
            if n_bytes <= 30:
                ax.set_xticks(range(n_bytes))
                ax.set_xticklabels([str(i+1) for i in range(n_bytes)], fontsize=8)
            
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=9, fontfamily='monospace')
            
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Normalized Position (0=start, 1=end, red=missing)', 
                          fontsize=9, color=SCIENCE_COLORS['text'])
            cbar.ax.tick_params(labelsize=8)
        
        elif plot_type == 'bar':
            ax = fig.add_subplot(111)
            ax.set_facecolor(SCIENCE_COLORS['background'])
            
            layer_data = []
            for layer_name in sorted(selected_layers, key=lambda x: self.layer_sizes.get(x, 0)):
                found = self.last_results.get(layer_name, {})
                hits = len(found)
                pct = (hits / total_bytes) * 100
                size = self.layer_sizes.get(layer_name, 0)
                
                if hits == total_bytes:
                    color = SCIENCE_COLORS['success']
                elif hits == total_bytes - 1:
                    color = SCIENCE_COLORS['primary']
                else:
                    color = SCIENCE_COLORS['warning']
                
                layer_data.append((layer_name, pct, size, color))
            
            x = range(len(layer_data))
            bars = ax.bar(x, [d[1] for d in layer_data], 
                         color=[d[3] for d in layer_data],
                         edgecolor=SCIENCE_COLORS['text'], linewidth=0.5, alpha=0.85)
            
            for i, (bar, data) in enumerate(zip(bars, layer_data)):
                height = bar.get_height()
                ax.annotate(f'{data[2]:,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=45,
                            color=SCIENCE_COLORS['text'])
            
            ax.set_xlabel('Layer (sorted by size)', fontsize=10,
                         color=SCIENCE_COLORS['text'], fontweight='medium')
            ax.set_ylabel('Match Rate (%)', fontsize=10,
                         color=SCIENCE_COLORS['text'], fontweight='medium')
            ax.set_title(f'Preimage Match Rate vs. Layer Size ({data_type})\nLabels indicate layer bit count', 
                         fontsize=11, fontweight='bold', color=SCIENCE_COLORS['text'], pad=15)
            
            ax.set_xticks(x)
            ax.set_xticklabels([d[0] for d in layer_data], rotation=45, ha='right', 
                              fontsize=8, fontfamily='monospace')
            ax.set_ylim(0, 105)
            
            ax.yaxis.grid(True, color=SCIENCE_COLORS['grid'], linestyle='-', linewidth=0.5)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            
            ax.axhline(y=80, color=SCIENCE_COLORS['found_marker'], linestyle='--', 
                      linewidth=1.2, alpha=0.7)
            
            legend_elements = [
                mpatches.Patch(facecolor=SCIENCE_COLORS['success'],
                              edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                              label='100% Match'),
                mpatches.Patch(facecolor=SCIENCE_COLORS['primary'],
                              edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                              label='N-1 Match'),
                mpatches.Patch(facecolor=SCIENCE_COLORS['warning'],
                              edgecolor=SCIENCE_COLORS['text'], linewidth=0.5,
                              label='<N-1 Match'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
                     edgecolor=SCIENCE_COLORS['grid'])
        
        fig.tight_layout()
        return fig
    
    def export_summary_csv(self, filepath, selected_layers):
        """Export summary statistics as CSV"""
        total_bytes = len(self.last_search_bytes)
        data_type = self.current_data_type.get()
        identifier = self.password_entry.get().strip()
        comment = self.comment_entry.get().strip()
        search_string = self.search_entry.get('1.0', tk.END).strip()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['GCIS Bitstring Search - Preprint Export'])
            writer.writerow(['Generated', datetime.now().isoformat()])
            writer.writerow(['Data Type', data_type])
            writer.writerow(['Identifier', identifier])
            writer.writerow(['Search String Length', f'{total_bytes} x {self.last_bit_length}-bit'])
            writer.writerow(['Comment', comment])
            writer.writerow([])
            
            # Search string
            writer.writerow(['SEARCH STRING (Binary)'])
            writer.writerow([search_string])
            writer.writerow([])
            
            # Summary table
            writer.writerow(['Layer', 'Layer_Bits', 'Matches', 'Match_Rate_%', 
                           'Coverage_%', 'Found_Positions'])
            
            for layer_name in sorted(selected_layers):
                found = self.last_results.get(layer_name, {})
                hits = len(found)
                size = self.layer_sizes.get(layer_name, 0)
                pct = (hits / total_bytes) * 100
                coverage = (hits * self.last_bit_length / size * 100) if size > 0 else 0
                
                # Positions string
                pos_list = []
                for byte in self.last_search_bytes:
                    if byte in found:
                        pos_list.append(str(found[byte][0]))
                    else:
                        pos_list.append('MISSING')
                
                writer.writerow([layer_name, size, hits, f'{pct:.2f}', 
                               f'{coverage:.4f}', ';'.join(pos_list)])
    
    def export_latex_table(self, filepath, selected_layers):
        """Export as LaTeX table"""
        total_bytes = len(self.last_search_bytes)
        data_type = self.current_data_type.get()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("% GCIS Bitstring Search - LaTeX Table Export\n")
            f.write(f"% Data Type: {data_type}\n")
            f.write(f"% Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Preimage Localization Results ({data_type})}}\n")
            f.write("\\label{tab:preimage_localization}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Layer & Bits & Matches & Match Rate (\\%) & Coverage (\\%) \\\\\n")
            f.write("\\midrule\n")
            
            for layer_name in sorted(selected_layers, key=lambda x: self.layer_sizes.get(x, 0)):
                found = self.last_results.get(layer_name, {})
                hits = len(found)
                size = self.layer_sizes.get(layer_name, 0)
                pct = (hits / total_bytes) * 100
                coverage = (hits * self.last_bit_length / size * 100) if size > 0 else 0
                
                # Escape underscores for LaTeX
                layer_tex = layer_name.replace('_', '\\_')
                
                f.write(f"{layer_tex} & {size:,} & {hits}/{total_bytes} & {pct:.1f} & {coverage:.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def export_raw_bitstrings(self, filepath, selected_layers):
        """Export raw bitstrings of selected layers"""
        data_type = self.current_data_type.get()
        identifier = self.password_entry.get().strip()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['GCIS Raw Bitstring Export'])
            writer.writerow(['Generated', datetime.now().isoformat()])
            writer.writerow(['Data Type', data_type])
            writer.writerow(['Identifier', identifier])
            writer.writerow([])
            
            # Bitstrings
            writer.writerow(['Layer', 'Bits', 'Bitstring'])
            
            for layer_name in sorted(selected_layers):
                bitstring = self.layers.get(layer_name, '')
                size = len(bitstring)
                writer.writerow([layer_name, size, bitstring])
    
    def open_search_export_dialog(self):
        """Opens the SearchResultsExportDialog for full CSV+TXT+PDF export"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        identifier = self.password_entry.get().strip() or "export"
        search_string = self.search_entry.get('1.0', tk.END).strip()
        
        dialog = SearchResultsExportDialog(
            self.root,
            results=self.last_results,
            search_bytes=self.last_search_bytes,
            layer_sizes=self.layer_sizes,
            data_type=self.current_data_type.get(),
            bit_length=self.last_bit_length,
            identifier=identifier,
            search_string=search_string,
            layers_data=self.layers
        )
    
    def export_csv_full(self):
        """Legacy simple CSV export"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        if not self.filepath:
            messagebox.showwarning("Warning", "No source directory!")
            return
        
        identifier = self.password_entry.get().strip()
        if not identifier:
            messagebox.showwarning("Warning", "Enter an identifier!")
            return
        
        total_bytes = len(self.last_search_bytes)
        
        full_match = [name for name, found in self.last_results.items() 
                      if len(found) == total_bytes]
        
        if not full_match:
            max_hits = max(len(found) for found in self.last_results.values())
            full_match = [name for name, found in self.last_results.items() 
                          if len(found) == max_hits]
        
        source_dir = os.path.dirname(self.filepath)
        safe_id = re.sub(r'[^\w\-]', '_', identifier)
        filepath = os.path.join(source_dir, f"{safe_id}_export.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Identifier', identifier])
            writer.writerow(['Data Type', self.current_data_type.get()])
            writer.writerow([])
            
            writer.writerow(['Layer', 'Bits', 'Matches', 'Percentage'])
            
            for layer_name in sorted(full_match):
                found = self.last_results.get(layer_name, {})
                size = self.layer_sizes.get(layer_name, 0)
                hits = len(found)
                pct = (hits / total_bytes) * 100
                writer.writerow([layer_name, size, f'{hits}/{total_bytes}', f'{pct:.1f}%'])
        
        messagebox.showinfo("Success", f"Exported: {filepath}")
    
    def analyze_missing(self):
        """Analyze missing bytes in N-1 layers"""
        if not self.last_results or not self.last_search_bytes:
            messagebox.showwarning("Warning", "Run a search first!")
            return
        
        self.missing_text.delete('1.0', tk.END)
        
        total_bytes = len(self.last_search_bytes)
        
        minus_one_layers = {}
        for layer_name, found_bytes in self.last_results.items():
            if len(found_bytes) == total_bytes - 1:
                minus_one_layers[layer_name] = found_bytes
        
        if not minus_one_layers:
            self.missing_text.insert(tk.END, "No layers with N-1 matches found.\n")
            return
        
        self.missing_text.insert(tk.END, "MISSING STRING ANALYSIS\n", 'red')
        self.missing_text.insert(tk.END, "=" * 60 + "\n\n")
        
        missing_stats = defaultdict(list)
        
        for layer_name, found_bytes in minus_one_layers.items():
            found_set = set(found_bytes.keys())
            missing_bytes = [b for b in self.last_search_bytes if b not in found_set]
            
            for missing in missing_bytes:
                missing_stats[missing].append(layer_name)
            
            self.missing_text.insert(tk.END, f"Layer: {layer_name}\n", 'blue')
            self.missing_text.insert(tk.END, f"  Found: {len(found_bytes)}/{total_bytes}\n")
            
            for missing in missing_bytes:
                char = self.byte_to_char(missing)
                idx = self.last_search_bytes.index(missing) + 1
                self.missing_text.insert(tk.END, f"  MISSING: Byte {idx} = {missing} '{char}'\n", 'red')
            
            self.missing_text.insert(tk.END, "\n")
        
        self.missing_text.insert(tk.END, "=" * 60 + "\n")
        self.missing_text.insert(tk.END, "SUMMARY: MISSING BYTES\n", 'gold')
        self.missing_text.insert(tk.END, "-" * 60 + "\n")
        
        for byte, layers in sorted(missing_stats.items(), key=lambda x: -len(x[1])):
            char = self.byte_to_char(byte)
            idx = self.last_search_bytes.index(byte) + 1
            self.missing_text.insert(tk.END, f"Byte {idx:2d} ({byte}) '{char}': missing in {len(layers)} layers\n")
    
    def byte_to_char(self, byte_str):
        """Convert binary string to character representation with ASCII decoding"""
        try:
            bit_len = len(byte_str)
            val = int(byte_str, 2)
            
            # 8-Bit: Direct ASCII character
            if bit_len == 8:
                if 32 <= val <= 126:
                    return chr(val)
                else:
                    return '.'
            
            # For bit lengths divisible by 8 (16, 24, 32): Show Hex + ASCII
            elif bit_len % 8 == 0 and 8 < bit_len <= 32:
                # Calculate hex representation
                hex_digits = bit_len // 4
                hex_str = format(val, f'0{hex_digits}X')
                
                # Decode ASCII characters (each 8 bits = 1 character)
                num_chars = bit_len // 8
                ascii_chars = []
                
                for i in range(num_chars):
                    # Extract each byte (8 bits)
                    byte_start = i * 8
                    byte_end = byte_start + 8
                    byte_bits = byte_str[byte_start:byte_end]
                    byte_val = int(byte_bits, 2)
                    
                    # Convert to ASCII if printable
                    if 32 <= byte_val <= 126:
                        ascii_chars.append(chr(byte_val))
                    else:
                        ascii_chars.append('.')
                
                ascii_str = ''.join(ascii_chars)
                return f"{hex_str} ({ascii_str})"
            
            # For other bit lengths: Just show Hex
            elif bit_len == 4:
                return format(val, 'X')
            elif bit_len == 12:
                return format(val, '03X')
            else:
                # Generic hex for any other length
                hex_digits = (bit_len + 3) // 4  # Round up
                return format(val, f'0{hex_digits}X')
        except:
            return '?'
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BitstringSearchGUI()
    app.run()