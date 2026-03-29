import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div className="flex items-center justify-center p-8">
            <div className="text-center">
              <div className="text-sm font-semibold text-red-400 mb-1">Something went wrong</div>
              <div className="text-xs font-mono text-red-500/70">{this.state.error.message}</div>
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
