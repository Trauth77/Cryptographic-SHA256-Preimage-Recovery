#!/usr/bin/env python3
"""
Cryptographic Hash & ECC Generator v3.0
- MD5, SHA-256, ECC-128 (secp128r1), ECC-256 (secp256k1)
- Automatic bit-length compatibility analysis (8-32 bit)
- Flexible binary/hex chunk output
- Auto-backup of last 25 passwords
- TXT export

Author: Stefan Trauth, 2025-2026
Part of: SHA-ECC-Decryption-Research-and-Results
"""

import hashlib
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from datetime import datetime
import json
import os

try:
    from ecdsa import SigningKey, SECP128r1, SECP256k1
    ECC_AVAILABLE = True
except ImportError:
    ECC_AVAILABLE = False


class CryptoGeneratorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cryptographic Hash & ECC Generator v3.0")
        self.root.geometry("1100x950")
        self.root.configure(bg='#FFFFFF')

        # --- Style constants ---
        self.BTN_BG = '#4374B3'
        self.BTN_FG = '#FFFFFF'
        self.BTN_FONT = ('Arial', 11, 'bold')

        # Selected bit length
        self.selected_bit_length = tk.IntVar(value=8)
        self.compatible_lengths = []

        # Selected crypto method
        self.selected_method = tk.StringVar(value='SHA-256')

        # History file path (same directory as script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.history_file = os.path.join(script_dir, "password_history.json")

        self.setup_gui()

    def setup_gui(self):
        # Header
        header = tk.Label(self.root, text="Cryptographic Hash & ECC Generator v3.0",
                          bg='#FFFFFF', font=('Arial', 16, 'bold'))
        header.pack(pady=15)

        # Input Frame
        input_frame = tk.Frame(self.root, bg='#FFFFFF')
        input_frame.pack(fill='x', padx=20)

        tk.Label(input_frame, text="Password / Input String:",
                 bg='#FFFFFF', font=('Arial', 12, 'bold')).pack(anchor='w')

        self.pw_entry = tk.Entry(input_frame, font=('Courier', 12))
        self.pw_entry.pack(fill='x', pady=5)
        self.pw_entry.bind('<KeyRelease>', self.on_password_change)

        # Method Selection Frame
        method_frame = tk.LabelFrame(self.root, text="Cryptographic Method",
                                     bg='#F8F8F8', font=('Arial', 11, 'bold'))
        method_frame.pack(fill='x', padx=20, pady=5)

        method_inner = tk.Frame(method_frame, bg='#F8F8F8')
        method_inner.pack(pady=10, padx=10)

        methods = ['MD5', 'SHA-256', 'ECC-128 (secp128r1)', 'ECC-256 (secp256k1)']
        for m in methods:
            tk.Radiobutton(method_inner, text=m, variable=self.selected_method,
                           value=m, bg='#F8F8F8', font=('Arial', 10),
                           command=self.on_method_change).pack(side='left', padx=15)

        if not ECC_AVAILABLE:
            tk.Label(method_frame,
                     text="⚠ ECC not available. Install: pip install ecdsa",
                     bg='#F8F8F8', font=('Arial', 9), fg='#DC3545').pack(pady=(0, 5))

        # Analysis Frame
        self.analysis_frame = tk.LabelFrame(self.root, text="Compatible Bit-Length Analysis",
                                            bg='#F0F8FF', font=('Arial', 11, 'bold'))
        self.analysis_frame.pack(fill='x', padx=20, pady=5)

        self.analysis_label = tk.Label(self.analysis_frame,
                                       text="Waiting for input...",
                                       bg='#F0F8FF', font=('Courier', 10),
                                       fg='#666666', justify='left')
        self.analysis_label.pack(pady=10, padx=10)

        # Bit Length Selection Frame
        selection_frame = tk.LabelFrame(self.root, text="Select Bit-Length for Output",
                                        bg='#F8F8F8', font=('Arial', 11, 'bold'))
        selection_frame.pack(fill='x', padx=20, pady=5)

        select_inner = tk.Frame(selection_frame, bg='#F8F8F8')
        select_inner.pack(pady=10, padx=10)

        tk.Label(select_inner, text="Bit-Length:",
                 bg='#F8F8F8', font=('Arial', 10, 'bold')).pack(side='left', padx=5)

        self.bit_length_combo = ttk.Combobox(select_inner,
                                             textvariable=self.selected_bit_length,
                                             state='readonly', width=10,
                                             font=('Arial', 10))
        self.bit_length_combo.pack(side='left', padx=5)
        self.bit_length_combo.bind('<<ComboboxSelected>>', self.on_bit_length_selected)

        tk.Label(select_inner,
                 text="Only compatible lengths available",
                 bg='#F8F8F8', font=('Arial', 9), fg='#666666').pack(side='left', padx=10)

        # Buttons
        btn_frame = tk.Frame(self.root, bg='#FFFFFF')
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="COMPUTE & SAVE", command=self.calculate_and_save,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=self.BTN_FONT,
                  width=18, cursor='hand2').pack(side='left', padx=8)

        tk.Button(btn_frame, text="CLEAR", command=self.clear,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=self.BTN_FONT,
                  width=12, cursor='hand2').pack(side='left', padx=8)

        tk.Button(btn_frame, text="EXPORT TXT", command=self.export_txt,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=self.BTN_FONT,
                  width=12, cursor='hand2').pack(side='left', padx=8)

        tk.Button(btn_frame, text="HISTORY", command=self.show_history,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=self.BTN_FONT,
                  width=12, cursor='hand2').pack(side='left', padx=8)

        # Output Area
        self.result_text = scrolledtext.ScrolledText(self.root, font=('Courier', 9), height=28)
        self.result_text.pack(padx=20, pady=10, fill='both', expand=True)

        self.result_text.insert(tk.END, "Enter a password above to start the analysis.\n")

    # ----------------------------------------------------------------
    # Crypto computations
    # ----------------------------------------------------------------

    def compute_md5(self, password):
        h = hashlib.md5(password.encode())
        return {'hex': h.hexdigest(), 'bytes': h.digest(), 'bits': 128, 'label': 'MD5 Hash (128-bit)'}

    def compute_sha256(self, password):
        h = hashlib.sha256(password.encode())
        return {'hex': h.hexdigest(), 'bytes': h.digest(), 'bits': 256, 'label': 'SHA-256 Hash (256-bit)'}

    def compute_ecc128(self, password):
        if not ECC_AVAILABLE:
            return None
        # Derive deterministic private key from password via SHA-256 truncated to 128 bit
        seed = hashlib.sha256(password.encode()).digest()[:16]  # 128 bit
        order = SECP128r1.order
        priv_int = int.from_bytes(seed, 'big') % (order - 1) + 1
        sk = SigningKey.from_secret_exponent(priv_int, curve=SECP128r1)
        vk = sk.get_verifying_key()
        pub_bytes = vk.to_string()  # uncompressed x+y coordinates
        sig = sk.sign_deterministic(password.encode())
        return {
            'private_key_hex': seed.hex(),
            'private_key_bytes': seed,
            'public_key_hex': pub_bytes.hex(),
            'public_key_bytes': pub_bytes,
            'signature_hex': sig.hex(),
            'signature_bytes': sig,
            'bits': 128,
            'label': 'ECC-128 / secp128r1 (ECDSA)'
        }

    def compute_ecc256(self, password):
        if not ECC_AVAILABLE:
            return None
        # Derive deterministic private key from password via SHA-256
        seed = hashlib.sha256(password.encode()).digest()  # 256 bit
        order = SECP256k1.order
        priv_int = int.from_bytes(seed, 'big') % (order - 1) + 1
        sk = SigningKey.from_secret_exponent(priv_int, curve=SECP256k1)
        vk = sk.get_verifying_key()
        pub_bytes = vk.to_string()
        sig = sk.sign_deterministic(password.encode())
        return {
            'private_key_hex': seed.hex(),
            'private_key_bytes': seed,
            'public_key_hex': pub_bytes.hex(),
            'public_key_bytes': pub_bytes,
            'signature_hex': sig.hex(),
            'signature_bytes': sig,
            'bits': 256,
            'label': 'ECC-256 / secp256k1 (ECDSA)'
        }

    # ----------------------------------------------------------------
    # Bit-length analysis
    # ----------------------------------------------------------------

    def get_output_bits(self):
        """Return the output bit size for current method"""
        method = self.selected_method.get()
        if method == 'MD5':
            return 128
        elif method == 'SHA-256':
            return 256
        elif method == 'ECC-128 (secp128r1)':
            return 128
        elif method == 'ECC-256 (secp256k1)':
            return 256
        return 256

    def find_compatible_bit_lengths(self, password):
        password_bits = len(password) * 8
        output_bits = self.get_output_bits()

        compatible = []
        for bit_length in range(8, 33):
            if password_bits % bit_length == 0 and output_bits % bit_length == 0:
                compatible.append(bit_length)
        return compatible

    def on_password_change(self, event=None):
        pw = self.pw_entry.get()

        if not pw:
            self.analysis_label.config(text="Waiting for input...", fg='#666666')
            self.bit_length_combo['values'] = []
            self.compatible_lengths = []
            return

        self.compatible_lengths = self.find_compatible_bit_lengths(pw)

        if self.compatible_lengths:
            pw_bits = len(pw) * 8
            output_bits = self.get_output_bits()
            method = self.selected_method.get()
            analysis_text = (f"Password: {len(pw)} chars = {pw_bits} bits\n"
                             f"{method}: {output_bits} bits\n"
                             f"Compatible: {', '.join(str(x) + '-bit' for x in self.compatible_lengths)}")
            self.analysis_label.config(text=analysis_text, fg='#059669')

            self.bit_length_combo['values'] = self.compatible_lengths
            if self.selected_bit_length.get() not in self.compatible_lengths:
                self.selected_bit_length.set(self.compatible_lengths[0])
        else:
            self.analysis_label.config(text="⚠ No compatible bit-lengths found!", fg='#DC3545')
            self.bit_length_combo['values'] = []

    def on_method_change(self):
        self.on_password_change()

    def on_bit_length_selected(self, event=None):
        if self.pw_entry.get():
            self.calculate_output()

    # ----------------------------------------------------------------
    # Binary / Hex helpers
    # ----------------------------------------------------------------

    def binary_to_chunks(self, binary_string, chunk_size):
        binary_clean = binary_string.replace(' ', '')
        chunks = [binary_clean[i:i + chunk_size] for i in range(0, len(binary_clean), chunk_size)]
        return ' '.join(chunks)

    def hex_to_chunks(self, hex_string, bit_length):
        hex_per_chunk = max(bit_length // 4, 1)
        chunks = [hex_string[i:i + hex_per_chunk] for i in range(0, len(hex_string), hex_per_chunk)]
        return ' '.join(chunks)

    def str_to_binary(self, text, bit_length=8):
        binary = ''.join(format(ord(c), '08b') for c in text)
        return self.binary_to_chunks(binary, bit_length)

    def bytes_to_binary(self, byte_data, bit_length=8):
        binary = ''.join(format(byte, '08b') for byte in byte_data)
        return self.binary_to_chunks(binary, bit_length)

    # ----------------------------------------------------------------
    # Output display
    # ----------------------------------------------------------------

    def calculate_output(self):
        pw = self.pw_entry.get()

        if not pw:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "Waiting for input...\n")
            return

        if not self.compatible_lengths:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "⚠ No compatible bit-lengths available!\n")
            return

        bit_len = self.selected_bit_length.get()
        method = self.selected_method.get()

        self.result_text.delete('1.0', tk.END)

        pw_binary = self.str_to_binary(pw, bit_len)

        self.result_text.insert(tk.END, f"{'=' * 80}\n")
        self.result_text.insert(tk.END, f"ANALYSIS — {method} — {bit_len}-bit chunks\n")
        self.result_text.insert(tk.END, f"{'=' * 80}\n\n")

        self.result_text.insert(tk.END, f"=== 1. PASSWORD (PLAINTEXT) ===\n")
        self.result_text.insert(tk.END, f"Text:    {pw}\n")
        self.result_text.insert(tk.END, f"Length:  {len(pw)} chars = {len(pw) * 8} bits\n")
        self.result_text.insert(tk.END, f"Binary:  {pw_binary}\n\n")

        if method == 'MD5':
            result = self.compute_md5(pw)
            self._display_hash_result(result, bit_len)

        elif method == 'SHA-256':
            result = self.compute_sha256(pw)
            self._display_hash_result(result, bit_len)

        elif method.startswith('ECC-128'):
            result = self.compute_ecc128(pw)
            if result is None:
                self.result_text.insert(tk.END, "⚠ ECC not available. Install: pip install ecdsa\n")
            else:
                self._display_ecc_result(result, bit_len)

        elif method.startswith('ECC-256'):
            result = self.compute_ecc256(pw)
            if result is None:
                self.result_text.insert(tk.END, "⚠ ECC not available. Install: pip install ecdsa\n")
            else:
                self._display_ecc_result(result, bit_len)

        self.result_text.insert(tk.END, f"\n{'=' * 80}\n")

    def _display_hash_result(self, result, bit_len):
        hex_spaced = self.hex_to_chunks(result['hex'], bit_len)
        binary = self.bytes_to_binary(result['bytes'], bit_len)

        self.result_text.insert(tk.END, f"=== 2. {result['label']} ===\n")
        self.result_text.insert(tk.END, f"Hex:           {result['hex']}\n")
        self.result_text.insert(tk.END, f"Hex (spaced):  {hex_spaced}\n")
        self.result_text.insert(tk.END, f"Binary:        {binary}\n")

    def _display_ecc_result(self, result, bit_len):
        self.result_text.insert(tk.END, f"=== 2. {result['label']} ===\n\n")

        # Private key
        priv_hex_spaced = self.hex_to_chunks(result['private_key_hex'], bit_len)
        priv_binary = self.bytes_to_binary(result['private_key_bytes'], bit_len)
        self.result_text.insert(tk.END, f"--- Private Key (derived from SHA-256 of password) ---\n")
        self.result_text.insert(tk.END, f"Hex:           {result['private_key_hex']}\n")
        self.result_text.insert(tk.END, f"Hex (spaced):  {priv_hex_spaced}\n")
        self.result_text.insert(tk.END, f"Binary:        {priv_binary}\n\n")

        # Public key
        pub_hex_spaced = self.hex_to_chunks(result['public_key_hex'], bit_len)
        pub_binary = self.bytes_to_binary(result['public_key_bytes'], bit_len)
        self.result_text.insert(tk.END, f"--- Public Key (x || y coordinates) ---\n")
        self.result_text.insert(tk.END, f"Hex:           {result['public_key_hex']}\n")
        self.result_text.insert(tk.END, f"Hex (spaced):  {pub_hex_spaced}\n")
        self.result_text.insert(tk.END, f"Binary:        {pub_binary}\n\n")

        # Signature
        sig_hex_spaced = self.hex_to_chunks(result['signature_hex'], bit_len)
        sig_binary = self.bytes_to_binary(result['signature_bytes'], bit_len)
        self.result_text.insert(tk.END, f"--- ECDSA Signature (deterministic, RFC 6979) ---\n")
        self.result_text.insert(tk.END, f"Hex:           {result['signature_hex']}\n")
        self.result_text.insert(tk.END, f"Hex (spaced):  {sig_hex_spaced}\n")
        self.result_text.insert(tk.END, f"Binary:        {sig_binary}\n")

    # ----------------------------------------------------------------
    # Save / History / Export
    # ----------------------------------------------------------------

    def calculate_and_save(self):
        pw = self.pw_entry.get()

        if not pw:
            messagebox.showwarning("Warning", "Please enter a password!")
            return

        if not self.compatible_lengths:
            messagebox.showwarning("Warning", "No compatible bit-lengths available!")
            return

        self.calculate_output()
        self.save_to_history(pw)

        messagebox.showinfo("Saved",
                            f"Password saved to history!\n"
                            f"File: {os.path.basename(self.history_file)}")

    def save_to_history(self, password):
        md5_hex = hashlib.md5(password.encode()).hexdigest()
        sha256_hex = hashlib.sha256(password.encode()).hexdigest()
        method = self.selected_method.get()

        entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'password': password,
            'length': len(password),
            'bits': len(password) * 8,
            'md5': md5_hex,
            'sha256': sha256_hex,
            'compatible_bit_lengths': self.compatible_lengths
        }

        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        history.insert(0, entry)
        history = history[:25]

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def show_history(self):
        if not os.path.exists(self.history_file):
            messagebox.showinfo("History", "No history yet.")
            return

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            messagebox.showerror("Error", "Could not load history!")
            return

        if not history:
            messagebox.showinfo("History", "History is empty.")
            return

        hist_window = tk.Toplevel(self.root)
        hist_window.title("Password History (Last 25)")
        hist_window.geometry("900x600")
        hist_window.configure(bg='#FFFFFF')

        tk.Label(hist_window, text="Password History",
                 bg='#FFFFFF', font=('Arial', 14, 'bold')).pack(pady=10)

        tk.Label(hist_window, text=f"Entries: {len(history)}",
                 bg='#FFFFFF', font=('Arial', 10)).pack()

        text_area = scrolledtext.ScrolledText(hist_window, font=('Courier', 9))
        text_area.pack(padx=20, pady=10, fill='both', expand=True)

        for i, entry in enumerate(history, 1):
            text_area.insert(tk.END, f"{'=' * 80}\n")
            text_area.insert(tk.END, f"#{i} | {entry['timestamp']} | {entry.get('method', 'N/A')}\n")
            text_area.insert(tk.END, f"{'=' * 80}\n")
            text_area.insert(tk.END, f"Password:    {entry['password']}\n")
            text_area.insert(tk.END, f"Length:      {entry['length']} chars ({entry['bits']} bits)\n")
            text_area.insert(tk.END, f"Compatible:  {', '.join(str(x) + '-bit' for x in entry.get('compatible_bit_lengths', []))}\n")
            text_area.insert(tk.END, f"MD5:         {entry['md5']}\n")
            text_area.insert(tk.END, f"SHA-256:     {entry['sha256']}\n\n")

        text_area.config(state='disabled')

        tk.Button(hist_window, text="Close", command=hist_window.destroy,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=('Arial', 10),
                  cursor='hand2').pack(pady=10)

    def export_txt(self):
        pw = self.pw_entry.get()

        if not pw:
            messagebox.showwarning("Warning", "Enter a password first!")
            return

        if not self.compatible_lengths:
            messagebox.showwarning("Warning", "No compatible bit-lengths!")
            return

        bit_len = self.selected_bit_length.get()
        method = self.selected_method.get()
        method_short = method.split(' ')[0].replace('-', '')

        filepath = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"crypto_report_{method_short}_{bit_len}bit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        if not filepath:
            return

        # Build report from current output
        content = self.result_text.get('1.0', tk.END)

        report = []
        report.append("=" * 80)
        report.append(f"CRYPTOGRAPHIC ANALYSIS REPORT")
        report.append(f"Method: {method} | Chunk Size: {bit_len}-bit")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Compatible Bit-Lengths: {', '.join(str(x) for x in self.compatible_lengths)}")
        report.append("=" * 80)
        report.append("")
        report.append(content)
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        messagebox.showinfo("Success", f"Report saved to:\n{filepath}")

    def clear(self):
        self.pw_entry.delete(0, tk.END)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "Ready.\n")
        self.analysis_label.config(text="Waiting for input...", fg='#666666')
        self.bit_length_combo['values'] = []
        self.compatible_lengths = []

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CryptoGeneratorGUI()
    app.run()