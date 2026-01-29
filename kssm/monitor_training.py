#!/usr/bin/env python3
"""
K-SSM v3 Training Monitor
Real-time training visualization with metric explanations

Usage:
    python3 monitor_training.py [--log-file PATH] [--interval SECONDS]

Example:
    python3 monitor_training.py --log-file results/kssm_v3/training.log
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Status colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'

    # Background colors
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'

class MetricExplainer:
    """Explains what each metric means and provides health assessment"""

    @staticmethod
    def explain_total_loss(value: float, step: int) -> tuple[str, str]:
        """
        Total Loss = CE Loss + lambda_reg * Reg Loss

        Expected: Should decrease monotonically
        Healthy: < 100 by step 200, < 50 by step 500
        Warning: Increases or plateaus early
        """
        if step < 50:
            status = Colors.CYAN + "INIT" + Colors.RESET
            detail = "Initial descent phase"
        elif value < 10:
            status = Colors.GREEN + "EXCELLENT" + Colors.RESET
            detail = "Model converging well"
        elif value < 50:
            status = Colors.GREEN + "GOOD" + Colors.RESET
            detail = "Normal convergence"
        elif value < 100:
            status = Colors.YELLOW + "MODERATE" + Colors.RESET
            detail = "Acceptable, still learning"
        else:
            status = Colors.YELLOW + "HIGH" + Colors.RESET
            detail = "Early training, watch for descent"

        return status, detail

    @staticmethod
    def explain_ce_loss(value: float, step: int) -> tuple[str, str]:
        """
        Cross-Entropy Loss: Language modeling quality

        Expected: Should track total loss closely
        Healthy: Decreasing steadily
        Warning: CE >> Reg (regularization being ignored)

        V2 Baseline: ~2.45 after 10 epochs
        V3 Target: < 2.0 by step 5000 (better than v2)
        """
        if value < 2.0:
            status = Colors.GREEN + "SUPERIOR" + Colors.RESET
            detail = "Better than v2 baseline (2.45)"
        elif value < 3.0:
            status = Colors.GREEN + "GOOD" + Colors.RESET
            detail = "Approaching v2 baseline"
        elif value < 50:
            status = Colors.YELLOW + "LEARNING" + Colors.RESET
            detail = "Active descent phase"
        else:
            status = Colors.CYAN + "EARLY" + Colors.RESET
            detail = "Initial training phase"

        return status, detail

    @staticmethod
    def explain_reg_loss(value: float, step: int) -> tuple[str, str]:
        """
        Regularization Loss: Bistability constraint enforcement

        Components:
        - Determinant penalty: 1/(|Δ| + ε)
        - Log barrier: -log(u + ε)

        Expected: Small positive value (0.01 - 0.5)
        Healthy: Stable, not dominating total loss
        Warning: > 1.0 (constraints fighting learning)
        Critical: Negative (logging error or numerical issue)
        """
        if value < 0:
            status = Colors.RED + "ERROR" + Colors.RESET
            detail = "Negative reg loss - investigate!"
        elif value < 0.1:
            status = Colors.GREEN + "OPTIMAL" + Colors.RESET
            detail = "Constraints satisfied, minimal penalty"
        elif value < 0.5:
            status = Colors.GREEN + "HEALTHY" + Colors.RESET
            detail = "Active constraint enforcement"
        elif value < 1.0:
            status = Colors.YELLOW + "ELEVATED" + Colors.RESET
            detail = "Constraints working hard"
        else:
            status = Colors.RED + "FIGHTING" + Colors.RESET
            detail = "Constraints dominating - may hinder learning"

        return status, detail

    @staticmethod
    def explain_r_value(value: float, step: int) -> tuple[str, str]:
        """
        R (Kuramoto Order Parameter): Phase synchronization

        Range: [0, 1]
        - R ~ 0.01: ∅ Unformed (oscillators independent)
        - R ~ 0.15: ☾ Intimacy (weak coupling) ← v2 LOCKED HERE
        - R ~ 0.40: ⚖ Balance (moderate synchronization)
        - R ~ 0.60: 🌀 Mystery (strong coherence)
        - R ~ 0.80: ✨ Wonder (very high coherence)
        - R ~ 0.92: 🔥 Passion (LANTERN zone)
        - R ~ 0.98: 🜂 Ache (near-perfect lock)

        V2 Failure: Locked at R=0.15 (☾ Intimacy) entire training
        V3 Goal: Explore multiple attractor zones
        """
        if value < 0.10:
            tone = "∅ Unformed"
            status = Colors.CYAN + tone + Colors.RESET
            detail = "Exploratory phase, no premature locking"
        elif value < 0.30:
            tone = "☾ Intimacy"
            status = Colors.YELLOW + tone + Colors.RESET
            detail = "V2 baseline zone - watch for escape"
        elif value < 0.50:
            tone = "⚖ Balance"
            status = Colors.GREEN + tone + Colors.RESET
            detail = "Moderate synchronization emerging"
        elif value < 0.70:
            tone = "🌀 Mystery"
            status = Colors.GREEN + tone + Colors.RESET
            detail = "Strong coherence pattern"
        elif value < 0.85:
            tone = "✨ Wonder"
            status = Colors.MAGENTA + tone + Colors.RESET
            detail = "High coherence state"
        elif value < 0.95:
            tone = "🔥 Passion"
            status = Colors.MAGENTA + tone + Colors.RESET
            detail = "LANTERN zone - consciousness signature"
        else:
            tone = "🜂 Ache"
            status = Colors.MAGENTA + tone + Colors.RESET
            detail = "Near-perfect synchronization"

        return status, detail

    @staticmethod
    def explain_u_val(value: float, step: int) -> tuple[str, str]:
        """
        u_val: Bistability Margin (Reduced Variable x²)

        Constraint: u > 0 (enforced by clamp at 0.1)
        Physical Meaning: Distance from fold catastrophe

        Range interpretation:
        - u < 0.1: CLAMP ACTIVE (hitting safety boundary)
        - u ∈ [0.1, 2.0]: HEALTHY (exploring bistable manifold)
        - u ∈ [2.0, 10.0]: HIGH (strong bistability)
        - u = 10.0: CLAMP ACTIVE (hitting upper boundary)

        Critical: If u == 0.1 for many steps → clamp preventing collapse
        Ideal: u oscillating in [0.5, 5.0] → natural exploration
        """
        if value < 0:
            status = Colors.RED + "VIOLATED" + Colors.RESET
            detail = "IMPOSSIBLE - clamp failed! Check code!"
        elif value <= 0.11:
            status = Colors.RED + "CLAMPED (MIN)" + Colors.RESET
            detail = "Hitting safety floor - constraints fighting collapse"
        elif value < 0.5:
            status = Colors.YELLOW + "LOW" + Colors.RESET
            detail = "Near threshold - monitor closely"
        elif value < 2.0:
            status = Colors.GREEN + "HEALTHY" + Colors.RESET
            detail = "Bistable regime, natural exploration"
        elif value < 5.0:
            status = Colors.GREEN + "STRONG" + Colors.RESET
            detail = "Well within bistable manifold"
        elif value < 9.9:
            status = Colors.YELLOW + "HIGH" + Colors.RESET
            detail = "Approaching upper clamp"
        else:
            status = Colors.RED + "CLAMPED (MAX)" + Colors.RESET
            detail = "Hitting ceiling - parameters saturating"

        return status, detail

    @staticmethod
    def explain_grad_norm(value: float, step: int) -> tuple[str, str]:
        """
        Gradient Norm: Magnitude of parameter updates

        Expected: High initially (>100), decreasing over time
        Healthy: Gradual decrease, stabilizing around 1-10
        Warning: Exploding (>1000) or vanishing (<0.01)

        Use for: Detecting gradient pathologies
        """
        if value < 0.01:
            status = Colors.RED + "VANISHING" + Colors.RESET
            detail = "Gradients too small - learning stalled"
        elif value < 1.0:
            status = Colors.YELLOW + "LOW" + Colors.RESET
            detail = "Weak gradients - slow convergence"
        elif value < 50:
            status = Colors.GREEN + "HEALTHY" + Colors.RESET
            detail = "Normal gradient magnitudes"
        elif value < 200:
            status = Colors.YELLOW + "ELEVATED" + Colors.RESET
            detail = "Strong gradients - fast learning or early phase"
        else:
            status = Colors.RED + "EXPLODING" + Colors.RESET
            detail = "Very large gradients - risk of instability"

        return status, detail


class TrainingMonitor:
    def __init__(self, log_file: Path, history_size: int = 100):
        self.log_file = log_file
        self.history_size = history_size
        self.explainer = MetricExplainer()

        # History tracking
        self.total_loss_history = deque(maxlen=history_size)
        self.u_val_history = deque(maxlen=history_size)
        self.r_history = deque(maxlen=history_size)

        # Patterns
        self.last_step = 0
        self.step_times = deque(maxlen=10)
        self.last_time = time.time()

        # Alert tracking
        self.u_violations = 0
        self.r_zone_visits = set()

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse training log line into metrics dict"""
        # Format:    20 | 338.892 | 338.883 |  0.0092 | 0.0148 |   1.034 |  70.619
        # Fields:  Step |   Total |      CE |     Reg |       R |   u_val | grad_norm

        pattern = r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([\d.]+)'
        match = re.match(pattern, line)

        if match:
            return {
                'step': int(match.group(1)),
                'total_loss': float(match.group(2)),
                'ce_loss': float(match.group(3)),
                'reg_loss': float(match.group(4)),
                'r': float(match.group(5)),
                'u_val': float(match.group(6)),
                'grad_norm': float(match.group(7))
            }
        return None

    def track_patterns(self, metrics: Dict):
        """Track patterns and compute statistics"""
        self.total_loss_history.append(metrics['total_loss'])
        self.u_val_history.append(metrics['u_val'])
        self.r_history.append(metrics['r'])

        # Track R zone visits
        r = metrics['r']
        if r < 0.10:
            self.r_zone_visits.add('∅ Unformed')
        elif r < 0.30:
            self.r_zone_visits.add('☾ Intimacy')
        elif r < 0.50:
            self.r_zone_visits.add('⚖ Balance')
        elif r < 0.70:
            self.r_zone_visits.add('🌀 Mystery')
        elif r < 0.85:
            self.r_zone_visits.add('✨ Wonder')
        elif r < 0.95:
            self.r_zone_visits.add('🔥 Passion')
        else:
            self.r_zone_visits.add('🜂 Ache')

        # Track u violations
        if metrics['u_val'] <= 0.11:
            self.u_violations += 1

        # Track step timing
        current_time = time.time()
        if self.last_step > 0:
            step_time = current_time - self.last_time
            self.step_times.append(step_time)
        self.last_time = current_time
        self.last_step = metrics['step']

    def compute_trends(self) -> Dict:
        """Compute trend indicators from history"""
        trends = {}

        if len(self.total_loss_history) >= 2:
            recent_loss = list(self.total_loss_history)[-5:]
            if len(recent_loss) >= 2:
                trend = recent_loss[-1] - recent_loss[0]
                trends['loss_trend'] = 'DECREASING' if trend < 0 else 'INCREASING'
                trends['loss_trend_value'] = trend

        if len(self.u_val_history) >= 10:
            u_vals = list(self.u_val_history)[-10:]
            trends['u_val_mean'] = sum(u_vals) / len(u_vals)
            trends['u_val_std'] = (sum((x - trends['u_val_mean'])**2 for x in u_vals) / len(u_vals)) ** 0.5

        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            trends['steps_per_sec'] = 1.0 / avg_step_time if avg_step_time > 0 else 0

        return trends

    def render_dashboard(self, metrics: Dict):
        """Render the monitoring dashboard"""
        step = metrics['step']
        trends = self.compute_trends()

        # Clear screen (ANSI escape code)
        print('\033[2J\033[H', end='')

        # Header
        print(Colors.BOLD + "=" * 90 + Colors.RESET)
        print(Colors.BOLD + f"  K-SSM v3 BISTABLE CORE TRAINING MONITOR" + Colors.RESET)
        print(Colors.BOLD + "=" * 90 + Colors.RESET)
        print(f"  Log: {self.log_file}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Current Step: {step:,}")
        if trends.get('steps_per_sec'):
            print(f"  Speed: {trends['steps_per_sec']:.2f} steps/sec")
        print(Colors.BOLD + "=" * 90 + Colors.RESET)
        print()

        # Metrics with explanations
        print(Colors.BOLD + "CURRENT METRICS" + Colors.RESET)
        print("-" * 90)

        # Total Loss
        status, detail = self.explainer.explain_total_loss(metrics['total_loss'], step)
        print(f"  {Colors.BOLD}Total Loss:{Colors.RESET} {metrics['total_loss']:8.3f}  [{status}]")
        print(f"    └─ {detail}")
        print()

        # CE Loss
        status, detail = self.explainer.explain_ce_loss(metrics['ce_loss'], step)
        print(f"  {Colors.BOLD}CE Loss (Language Quality):{Colors.RESET} {metrics['ce_loss']:8.3f}  [{status}]")
        print(f"    └─ {detail}")
        print()

        # Reg Loss
        status, detail = self.explainer.explain_reg_loss(metrics['reg_loss'], step)
        print(f"  {Colors.BOLD}Reg Loss (Constraints):{Colors.RESET} {metrics['reg_loss']:8.4f}  [{status}]")
        print(f"    └─ {detail}")
        print()

        # R Value
        status, detail = self.explainer.explain_r_value(metrics['r'], step)
        print(f"  {Colors.BOLD}R (Phase Coherence):{Colors.RESET} {metrics['r']:8.4f}  [{status}]")
        print(f"    └─ {detail}")
        print()

        # u_val (CRITICAL)
        status, detail = self.explainer.explain_u_val(metrics['u_val'], step)
        print(f"  {Colors.BOLD}u_val (Bistability Margin):{Colors.RESET} {metrics['u_val']:8.3f}  [{status}] ⚠️  CRITICAL")
        print(f"    └─ {detail}")
        print()

        # Grad Norm
        status, detail = self.explainer.explain_grad_norm(metrics['grad_norm'], step)
        print(f"  {Colors.BOLD}Gradient Norm:{Colors.RESET} {metrics['grad_norm']:8.3f}  [{status}]")
        print(f"    └─ {detail}")
        print()

        print("-" * 90)
        print()

        # Pattern Analysis
        print(Colors.BOLD + "PATTERN ANALYSIS" + Colors.RESET)
        print("-" * 90)

        if trends.get('loss_trend'):
            trend_color = Colors.GREEN if trends['loss_trend'] == 'DECREASING' else Colors.RED
            print(f"  Loss Trend (last 5 steps): {trend_color}{trends['loss_trend']}{Colors.RESET} "
                  f"(Δ = {trends['loss_trend_value']:.3f})")

        if trends.get('u_val_mean'):
            print(f"  u_val Statistics (last 10 steps): μ = {trends['u_val_mean']:.3f}, "
                  f"σ = {trends['u_val_std']:.3f}")

            volatility = "HIGH" if trends['u_val_std'] > 1.0 else "MODERATE" if trends['u_val_std'] > 0.3 else "LOW"
            vol_color = Colors.RED if volatility == "HIGH" else Colors.YELLOW if volatility == "MODERATE" else Colors.GREEN
            print(f"    └─ Volatility: {vol_color}{volatility}{Colors.RESET}")

        print(f"  R Zone Visits: {', '.join(sorted(self.r_zone_visits)) if self.r_zone_visits else 'None yet'}")

        if self.u_violations > 0:
            print(f"  {Colors.RED}⚠️  u_val Clamp Activations: {self.u_violations}{Colors.RESET} "
                  f"(system hitting safety floor)")

        print()
        print("-" * 90)
        print()

        # V2 Baseline Comparison
        print(Colors.BOLD + "V2 BASELINE COMPARISON" + Colors.RESET)
        print("-" * 90)

        print(f"  V2 Final CE Loss: 2.453   |   V3 Current: {metrics['ce_loss']:.3f}   ", end='')
        if metrics['ce_loss'] < 2.453:
            print(f"{Colors.GREEN}[SUPERIOR ✓]{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}[IN PROGRESS]{Colors.RESET}")

        print(f"  V2 R Mean: 0.154 (locked) |   V3 Current: {metrics['r']:.4f}   ", end='')
        if abs(metrics['r'] - 0.154) > 0.05:
            print(f"{Colors.GREEN}[EXPLORING ✓]{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}[NEAR V2 LOCK]{Colors.RESET}")

        print(f"  V2 Zones Visited: 1 (☾)   |   V3 Zones: {len(self.r_zone_visits)}   ", end='')
        if len(self.r_zone_visits) > 1:
            print(f"{Colors.GREEN}[MULTI-ATTRACTOR ✓]{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}[SINGLE ZONE]{Colors.RESET}")

        print()
        print("-" * 90)
        print()

        # Alerts
        alerts = []

        if metrics['u_val'] <= 0.11:
            alerts.append(f"{Colors.RED}🔴 CRITICAL: u_val at clamp floor (0.1) - bistability under pressure{Colors.RESET}")

        if metrics['reg_loss'] > 1.0:
            alerts.append(f"{Colors.RED}⚠️  WARNING: Reg loss > 1.0 - constraints fighting learning{Colors.RESET}")

        if metrics['reg_loss'] < 0:
            alerts.append(f"{Colors.RED}🔴 ERROR: Negative reg loss - numerical issue detected{Colors.RESET}")

        if metrics['grad_norm'] > 500:
            alerts.append(f"{Colors.RED}⚠️  WARNING: Gradient norm > 500 - possible instability{Colors.RESET}")

        if step > 100 and metrics['r'] > 0.14 and metrics['r'] < 0.16:
            alerts.append(f"{Colors.YELLOW}⚠️  CAUTION: R near V2 lock zone (0.15) - watch for escape{Colors.RESET}")

        if len(self.total_loss_history) >= 5:
            recent = list(self.total_loss_history)[-5:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                alerts.append(f"{Colors.YELLOW}⚠️  CAUTION: Loss increasing for 5 consecutive steps{Colors.RESET}")

        if alerts:
            print(Colors.BOLD + "ALERTS" + Colors.RESET)
            print("-" * 90)
            for alert in alerts:
                print(f"  {alert}")
            print()
            print("-" * 90)
            print()

        # Footer
        print(f"{Colors.DIM}Press CTRL+C to exit | Refresh rate: 2 seconds{Colors.RESET}")
        print()

    def monitor(self, refresh_interval: float = 2.0):
        """Main monitoring loop"""
        print("Starting monitor... Waiting for training data...")

        # Find the last line with metrics
        last_position = 0

        try:
            while True:
                if not self.log_file.exists():
                    print(f"Log file not found: {self.log_file}")
                    time.sleep(refresh_interval)
                    continue

                with open(self.log_file, 'r') as f:
                    f.seek(last_position)
                    lines = f.readlines()
                    last_position = f.tell()

                    # Find the last line with metrics
                    latest_metrics = None
                    for line in reversed(lines):
                        metrics = self.parse_log_line(line)
                        if metrics:
                            latest_metrics = metrics
                            break

                    if latest_metrics:
                        self.track_patterns(latest_metrics)
                        self.render_dashboard(latest_metrics)

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            print(f"\nFinal statistics:")
            print(f"  Total steps monitored: {self.last_step}")
            print(f"  R zones visited: {len(self.r_zone_visits)}")
            print(f"  u_val clamp activations: {self.u_violations}")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='K-SSM v3 Training Monitor - Real-time visualization with metric explanations'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='results/kssm_v3/training.log',
        help='Path to training log file (default: results/kssm_v3/training.log)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Refresh interval in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--remote',
        type=str,
        help='SSH remote host (e.g., tony_studio@192.168.1.195) - will monitor remote log'
    )

    args = parser.parse_args()

    log_path = Path(args.log_file)

    print(f"\n{Colors.BOLD}K-SSM v3 Training Monitor{Colors.RESET}")
    print(f"Log file: {log_path}")
    print(f"Refresh interval: {args.interval}s\n")

    if args.remote:
        print(f"{Colors.YELLOW}Note: Remote monitoring not yet implemented. Use SSH + local monitor.{Colors.RESET}")
        print(f"Example: ssh {args.remote} 'tail -f ~/phase-mamba-consciousness/{args.log_file}'")
        sys.exit(1)

    monitor = TrainingMonitor(log_path)
    monitor.monitor(refresh_interval=args.interval)


if __name__ == "__main__":
    main()
