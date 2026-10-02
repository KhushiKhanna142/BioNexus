import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '@/lib/utils';

// ============== Stat Card ==============

interface StatCardProps {
  icon: React.ReactNode;
  value: string | number;
  label: string;
  sublabel?: string;
  className?: string;
}

export function StatCard({ icon, value, label, sublabel, className }: StatCardProps) {
  const shouldReduce = useReducedMotion();

  return (
    <motion.div
      initial={shouldReduce ? {} : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={cn('bio-card p-5 flex items-start gap-4', className)}
    >
      <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-primary)] flex items-center justify-center shrink-0 text-[var(--color-accent)]">
        {icon}
      </div>
      <div>
        <AnimatedNumber value={value} className="text-2xl font-bold text-[var(--color-navy)]" />
        <p className="text-sm text-[var(--color-text-secondary)] mt-0.5">{label}</p>
        {sublabel && (
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5">{sublabel}</p>
        )}
      </div>
    </motion.div>
  );
}

// ============== Animated Number ==============

interface AnimatedNumberProps {
  value: string | number;
  className?: string;
}

export function AnimatedNumber({ value, className }: AnimatedNumberProps) {
  const shouldReduce = useReducedMotion();
  const [displayed, setDisplayed] = useState(shouldReduce ? value : typeof value === 'number' ? 0 : value);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (shouldReduce || typeof value !== 'number') {
      setDisplayed(value);
      return;
    }
    const duration = 1200;
    const start = performance.now();
    const startVal = 0;
    const animate = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (value - startVal) * eased;
      setDisplayed(Number.isInteger(value) ? Math.round(current) : parseFloat(current.toFixed(1)));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [value, shouldReduce]);

  return <span ref={ref} className={className}>{displayed}{typeof value === 'string' && ''}</span>;
}

// ============== Section Card ==============

interface SectionCardProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export function SectionCard({ title, subtitle, badge, actions, children, className }: SectionCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn('bio-card p-6', className)}
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-[var(--color-navy)]">{title}</h2>
            {badge}
          </div>
          {subtitle && (
            <p className="text-sm text-[var(--color-text-muted)] mt-0.5">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </motion.div>
  );
}

// ============== Tab Pills ==============

interface TabPillsProps {
  tabs: { id: string; label: string; disabled?: boolean; badge?: string }[];
  activeTab: string;
  onChange: (id: string) => void;
}

export function TabPills({ tabs, activeTab, onChange }: TabPillsProps) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap" role="tablist">
      {tabs.map(tab => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={activeTab === tab.id}
          disabled={tab.disabled}
          onClick={() => !tab.disabled && onChange(tab.id)}
          className={cn(
            'tab-pill',
            activeTab === tab.id ? 'tab-pill-active' : 'tab-pill-inactive',
            tab.disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          {tab.label}
          {tab.badge && (
            <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-[rgba(139,111,224,0.15)] text-[var(--color-basal)]">
              {tab.badge}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// ============== Skeleton ==============

interface SkeletonProps {
  className?: string;
  lines?: number;
}

export function Skeleton({ className, lines = 1 }: SkeletonProps) {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="skeleton h-4"
          style={{ width: i === lines - 1 ? '60%' : '100%' }}
        />
      ))}
    </div>
  );
}

export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('bio-card p-6', className)}>
      <Skeleton className="mb-4" />
      <Skeleton lines={3} />
    </div>
  );
}

// ============== Progress Bar ==============

interface ProgressBarProps {
  value: number;
  max?: number;
  color?: string;
  label?: string;
  showValue?: boolean;
  className?: string;
  height?: string;
}

export function ProgressBar({ value, max = 1, color = 'var(--color-accent)', label, showValue = true, className, height = '8px' }: ProgressBarProps) {
  const percent = Math.min((value / max) * 100, 100);

  return (
    <div className={cn('w-full', className)}>
      {(label || showValue) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>}
          {showValue && <span className="text-sm font-medium tabular-nums">{value.toFixed(2)}</span>}
        </div>
      )}
      <div className="w-full bg-[var(--color-bg-primary)] rounded-full overflow-hidden" style={{ height }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
    </div>
  );
}


// ============== Research Disclaimer ==============

export function ResearchDisclaimer() {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700">
      <span className="font-medium">⚠️ Research use only</span>
      <span>— Not for clinical decisions.</span>
    </div>
  );
}

// ============== Glossary Tooltip ==============

import { GLOSSARY } from '@/lib/utils';

interface GlossaryTermProps {
  term: string;
  children?: React.ReactNode;
}

export function GlossaryTerm({ term, children }: GlossaryTermProps) {
  const definition = GLOSSARY[term];
  if (!definition) return <span>{children || term}</span>;

  return (
    <span className="relative group inline-block">
      <span className="border-b border-dashed border-[var(--color-text-muted)] cursor-help">
        {children || term}
      </span>
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-[var(--color-navy)] text-white text-xs rounded-lg shadow-lg opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity z-50">
        <strong className="block mb-1">{term}</strong>
        {definition}
      </span>
    </span>
  );
}

// ============== Empty State ==============

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {icon && <div className="w-12 h-12 rounded-full bg-[var(--color-bg-primary)] flex items-center justify-center text-[var(--color-text-muted)] mb-4">{icon}</div>}
      <h3 className="text-lg font-semibold text-[var(--color-navy)] mb-1">{title}</h3>
      <p className="text-sm text-[var(--color-text-muted)] max-w-sm mb-4">{description}</p>
      {action}
    </div>
  );
}

// ============== Error State ==============

interface ErrorStateProps {
  error: Error | string;
  onRetry?: () => void;
}

export function ErrorState({ error, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center text-red-500 mb-4 text-xl">!</div>
      <h3 className="text-lg font-semibold text-[var(--color-navy)] mb-1">Something went wrong</h3>
      <p className="text-sm text-[var(--color-text-muted)] max-w-sm mb-4">
        {typeof error === 'string' ? error : error.message}
      </p>
      {onRetry && (
        <button onClick={onRetry} className="btn-primary text-sm">
          Try again
        </button>
      )}
    </div>
  );
}
