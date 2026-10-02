import { TopBar } from '@/components/layout/TopBar';
import { SectionCard } from '@/components/ui';
import { Lightbulb, AlertTriangle, Layers, BookOpen } from 'lucide-react';

export function PathwayAnalysisPage() {
  return (
    <div>
      <TopBar
        title="Pathway Analysis"
        subtitle="Biological pathway enrichment for multi-omics driver signatures."
      />
      <div className="p-6 space-y-6 max-w-4xl">
        {/* Planned Feature Banner */}
        <div className="flex items-center gap-3 p-4 bg-purple-50 border border-purple-200 rounded-2xl text-purple-900 text-xs">
          <div className="w-8 h-8 rounded-xl bg-purple-100 flex items-center justify-center shrink-0 text-purple-700">
            <Lightbulb className="w-4 h-4" />
          </div>
          <div>
            <span className="font-bold text-sm block">Status: Planned Feature (Disabled)</span>
            <p className="text-purple-700 mt-0.5 leading-relaxed">
              Curated gene set collections (MSigDB Hallmark / KEGG <code className="font-mono bg-purple-100 px-1 py-0.5 rounded text-[11px]">.gmt</code> files)
              were not supplied in the local repository artifacts. In compliance with strict zero-mock rules,
              synthetic or illustrative pathway charts are completely disabled until external annotation sets are configured.
            </p>
          </div>
        </div>

        <SectionCard title="Planned Pathway Enrichment Specifications">
          <div className="space-y-4 text-xs text-slate-600">
            <p className="leading-relaxed">
              When external gene set databases are integrated, this module will execute Fisher's exact / hypergeometric over-representation tests
              against the top-attributed genes identified by Integrated Gradients for each patient and PAM50 subtype:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-1">
                <span className="font-bold text-slate-900 block">Biological Pathways (MSigDB)</span>
                <p className="text-slate-500 text-[11px]">
                  50 Hallmark gene sets representing well-defined biological states and processes (e.g. Estrogen Response, DNA Repair, G2M Checkpoint).
                </p>
              </div>

              <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-1">
                <span className="font-bold text-slate-900 block">Gene Ontology (GO Biological Process)</span>
                <p className="text-slate-500 text-[11px]">
                  Over-representation analysis with Benjamini-Hochberg false discovery rate (FDR &lt; 0.05) correction across hierarchical GO terms.
                </p>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
