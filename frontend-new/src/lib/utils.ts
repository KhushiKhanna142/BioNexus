import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

/**
 * Glossary terms for hover cards
 */
export const GLOSSARY: Record<string, string> = {
  'PAM50': 'A 50-gene signature used to classify breast cancer into five molecular subtypes: Luminal A, Luminal B, HER2-enriched, Basal-like, and Normal-like.',
  'Omics': 'Large-scale biological data types such as genomics, transcriptomics (RNA), and epigenomics (DNA methylation).',
  'Methylation': 'A biochemical process involving addition of a methyl group to DNA, which can regulate gene expression without changing the DNA sequence.',
  'miRNA': 'MicroRNA — small non-coding RNA molecules that regulate gene expression post-transcriptionally.',
  'Logit': 'The raw output of a neural network before applying softmax to convert to probabilities.',
  'Ablation': 'A systematic study where components are removed to assess their contribution to model performance.',
  'GraphGPS': 'A graph transformer architecture that combines message-passing with global attention for learning on graph-structured data.',
  'Integrated Gradients': 'An attribution method that assigns importance scores to input features by integrating gradients along a path from a baseline to the input.',
  'Cross-attention': 'A mechanism that allows one data modality to attend to and integrate information from another modality.',
  'TCGA': 'The Cancer Genome Atlas — a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples.',
  'kNN': 'k-Nearest Neighbors — a method that connects each data point to its k most similar neighbors, used here to build patient similarity graphs.',
};
