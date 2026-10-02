import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, UserSearch, Database, Brain,
  Users, BookOpen, Settings, ChevronLeft, ChevronRight,
  FlaskConical, Route, Share2,
} from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/patient-analysis', label: 'Patient Analysis', icon: UserSearch },
  { path: '/graph-explorer', label: 'Graph Explorer', icon: Share2 },
  { path: '/data-explorer', label: 'Data Explorer', icon: Database },
  { path: '/model-insights', label: 'Model Insights', icon: Brain },
  { path: '/cohort-analysis', label: 'Cohort Analysis', icon: Users },
  { path: '/research', label: 'Research', icon: BookOpen },
];

const SECONDARY_ITEMS = [
  { path: '/biomarkers', label: 'Biomarkers', icon: FlaskConical },
  { path: '/pathway-analysis', label: 'Pathway Analysis', icon: Route },
];

const BOTTOM_ITEMS = [
  { path: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-screen z-50 flex flex-col bg-white border-r border-slate-100 transition-all duration-300 shadow-[1px_0_4px_rgba(0,0,0,0.02)]',
        collapsed ? 'w-[72px]' : 'w-[230px]',
        'max-lg:w-[72px] max-md:hidden'
      )}
    >
      {/* Brand Logo */}
      <div className={cn(
        'flex items-center gap-3 px-5 h-18 border-b border-slate-100 shrink-0',
        collapsed && 'justify-center px-0'
      )}>
        {/* Custom molecular / graph network logo symbol */}
        <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shrink-0 shadow-md shadow-blue-500/25">
          <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="6" cy="6" r="2.5" />
            <circle cx="18" cy="6" r="2.5" />
            <circle cx="12" cy="12" r="3" />
            <circle cx="6" cy="18" r="2.5" />
            <circle cx="18" cy="18" r="2.5" />
            <line x1="8" y1="7" x2="10" y2="10" />
            <line x1="16" y1="7" x2="14" y2="10" />
            <line x1="10" y1="14" x2="8" y2="17" />
            <line x1="14" y1="14" x2="16" y2="17" />
          </svg>
        </div>
        {!collapsed && (
          <div className="flex flex-col">
            <span className="font-bold text-lg text-slate-900 tracking-tight leading-none">
              BioNexus
            </span>
            <span className="text-[10px] text-slate-400 font-medium tracking-wide uppercase mt-0.5">
              Multi-Omics ML
            </span>
          </div>
        )}
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 py-4 px-3 overflow-y-auto space-y-1" aria-label="Main navigation">
        <div className="space-y-1">
          {NAV_ITEMS.map(item => {
            const isActive = item.path === '/' 
              ? location.pathname === '/' 
              : location.pathname.startsWith(item.path);

            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={cn(
                  'flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-[13.5px] font-medium transition-all duration-150',
                  isActive
                    ? 'bg-blue-50 text-blue-600 font-semibold shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50',
                  collapsed && 'justify-center px-0'
                )}
                aria-current={isActive ? 'page' : undefined}
                title={item.label}
              >
                <item.icon className={cn('w-4.5 h-4.5 shrink-0', isActive ? 'text-blue-600' : 'text-slate-500')} />
                {!collapsed && <span className="max-lg:hidden">{item.label}</span>}
              </NavLink>
            );
          })}
        </div>

        {/* Secondary Navigation */}
        <div className="pt-3 mt-3 border-t border-slate-100">
          {!collapsed && (
            <p className="px-3 pb-1.5 text-[10px] uppercase tracking-wider font-semibold text-slate-400 max-lg:hidden">
              Deep Dives
            </p>
          )}
          {SECONDARY_ITEMS.map(item => {
            const isActive = location.pathname.startsWith(item.path);
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={cn(
                  'flex items-center gap-3 px-3.5 py-2 rounded-xl text-[13px] font-medium transition-all duration-150',
                  isActive
                    ? 'bg-blue-50 text-blue-600 font-semibold'
                    : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50',
                  collapsed && 'justify-center px-0'
                )}
                title={item.label}
              >
                <item.icon className={cn('w-4 h-4 shrink-0', isActive ? 'text-blue-600' : 'text-slate-400')} />
                {!collapsed && <span className="max-lg:hidden">{item.label}</span>}
              </NavLink>
            );
          })}
        </div>
      </nav>

      {/* Bottom Nav: Settings & Collapse */}
      <div className="py-3 px-3 border-t border-slate-100 space-y-1">
        {BOTTOM_ITEMS.map(item => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={cn(
                'flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-[13.5px] font-medium transition-all duration-150',
                isActive ? 'bg-blue-50 text-blue-600 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50',
                collapsed && 'justify-center px-0'
              )}
              title={item.label}
            >
              <item.icon className="w-4.5 h-4.5 shrink-0 text-slate-500" />
              {!collapsed && <span className="max-lg:hidden">{item.label}</span>}
            </NavLink>
          );
        })}

        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            'flex items-center gap-3 w-full px-3.5 py-2 text-[12px] font-medium text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-xl transition-colors',
            collapsed && 'justify-center px-0'
          )}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span className="max-lg:hidden">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
