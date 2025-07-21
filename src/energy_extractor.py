#!/usr/bin/env python3
"""
energy_extractor.py - Extract energies from all ORCA .out files in a folder

This script processes all .out files in a selected folder, extracts Gibbs free energies
(or falls back to single point energies), and creates a CSV file with the results.

Usage:
    python energy_extractor.py

The script will open a file dialog to select the folder containing .out files.
Output: energies.csv with columns: filename, energy_hartree, energy_kcal_mol

Author: AI Assistant (2025)
"""

import os
import re
import sys
import glob
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pathlib import Path

# Conversion factor from Hartree to kcal/mol
HARTREE_TO_KCAL = 627.509


def select_folder():
    """Open a file dialog to select a folder containing .out files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    folder_path = filedialog.askdirectory(
        title="Select folder containing ORCA .out files"
    )

    root.destroy()
    return folder_path


def extract_energy(file_path, verbose=False):
    """
    Extract Gibbs free energy or single point energy from an ORCA .out file.

    Args:
        file_path (str): Path to the .out file
        verbose (bool): Print debug information

    Returns:
        float: Energy in Hartree, or None if not found
    """
    # Primary patterns for Gibbs free energy
    gibbs_patterns = [
        re.compile(r"Total\s+Gibbs\s+Free\s+Energy\s*:\s*([-]?\d+\.\d+)"),
        re.compile(r"TOTAL\s+FREE\s+ENERGY\s*:\s*([-]?\d+\.\d+)"),
        re.compile(r"FINAL\s+GIBBS\s+FREE\s+ENERGY\s*:\s*([-]?\d+\.\d+)"),
        re.compile(r"Gibbs\s+free\s+energy\s*\(.*\)\s*=\s*([-]?\d+\.\d+)\s*Eh", re.I),
        re.compile(r"Final\s+Gibbs\s+free\s+energy.*?([-]?\d+\.\d+)\s*Eh?", re.I),
    ]

    # Fallback patterns for single point energy
    sp_patterns = [
        re.compile(r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s*:\s*([-]?\d+\.\d+)", re.I),
        re.compile(r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-]?\d+\.\d+)", re.I),
        re.compile(r"SCF\s+TOTAL\s+ENERGY\s*:\s*([-]?\d+\.\d+)", re.I),
    ]

    try:
        with open(file_path, "r", errors="ignore") as fh:
            # Read file from bottom up since energy values are usually at the end
            lines = fh.readlines()[::-1]

            # First try to find Gibbs free energy
            for line in lines:
                for pattern in gibbs_patterns:
                    match = pattern.search(line)
                    if match:
                        energy = float(match.group(1))
                        if verbose:
                            print(f"[GIBBS] {os.path.basename(file_path)}: {line.strip()}")
                        return energy

            # If no Gibbs energy found, try single point energy
            for line in lines:
                for pattern in sp_patterns:
                    match = pattern.search(line)
                    if match:
                        energy = float(match.group(1))
                        if verbose:
                            print(f"[SP] {os.path.basename(file_path)}: {line.strip()}")
                            print("     (No Gibbs energy found; using single point energy)")
                        return energy

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to read {file_path}: {e}")
        return None

    if verbose:
        print(f"[WARN] No energy found in {os.path.basename(file_path)}")
    return None


def process_folder(folder_path, verbose=False):
    """
    Process all .out files in the given folder and extract energies.

    Args:
        folder_path (str): Path to folder containing .out files
        verbose (bool): Print debug information

    Returns:
        list: List of dictionaries with filename and energy data
    """
    if not os.path.isdir(folder_path):
        print(f"ERROR: {folder_path} is not a valid directory")
        return []

    # Find all .out files (case insensitive)
    out_files = []
    for pattern in ("*.out", "*.OUT"):
        out_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    if not out_files:
        print(f"ERROR: No .out files found in {folder_path}")
        return []

    print(f"Found {len(out_files)} .out files to process...")

    results = []
    for file_path in sorted(out_files):
        filename = os.path.basename(file_path)

        if verbose:
            print(f"\nProcessing: {filename}")

        energy_hartree = extract_energy(file_path, verbose)

        if energy_hartree is not None:
            energy_kcal = energy_hartree * HARTREE_TO_KCAL
            results.append({
                'filename': filename,
                'energy_hartree': energy_hartree,
                'energy_kcal_mol': energy_kcal
            })

            if not verbose:
                print(f"✓ {filename}: {energy_hartree:.6f} Eh ({energy_kcal:.2f} kcal/mol)")
        else:
            print(f"✗ {filename}: No energy found")
            results.append({
                'filename': filename,
                'energy_hartree': None,
                'energy_kcal_mol': None
            })

    return results


def save_to_csv(results, output_path):
    """
    Save the results to a CSV file.

    Args:
        results (list): List of dictionaries with energy data
        output_path (str): Path where to save the CSV file
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main function to run the energy extraction process."""
    print("ORCA Energy Extractor")
    print("=" * 50)

    # Get folder from user
    folder_path = select_folder()

    if not folder_path:
        print("No folder selected. Exiting.")
        sys.exit(1)

    print(f"Selected folder: {folder_path}")

    # Ask for verbose output
    verbose_input = input("\nVerbose output? (y/n): ").lower().strip()
    verbose = verbose_input.startswith('y')

    # Process all .out files
    results = process_folder(folder_path, verbose=verbose)

    if not results:
        print("No files were processed successfully.")
        sys.exit(1)

    # Save results to CSV in the same folder
    output_path = os.path.join(folder_path, "energies.csv")
    save_to_csv(results, output_path)

    # Summary
    successful = sum(1 for r in results if r['energy_hartree'] is not None)
    total = len(results)
    print(f"\nSummary: {successful}/{total} files processed successfully")

    if successful < total:
        failed_files = [r['filename'] for r in results if r['energy_hartree'] is None]
        print(f"Failed files: {', '.join(failed_files)}")


if __name__ == "__main__":
    main()