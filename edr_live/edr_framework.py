"""
edr_framework.py
────────────────────────────────────────────────────────
Emotion Disagreement Resolution (EDR) Framework

Authors : Md Adil     (22SCSE1012699)
          Rohit Kumar Ram (22SCSE1012599)
College : Galgotias University, Greater Noida

Algorithm (from paper Section III-D):
  1. If Ef == Es          → AGREEMENT  (output directly)
  2. Compute Δ = |Cf - Cs|
  3. If Δ > τ             → DOMINANCE  (higher-confidence wins)
  4. If Δ ≤ τ             → UNCERTAIN  (sarcasm / mixed emotion)
────────────────────────────────────────────────────────
"""


class EDRFramework:
    """Confidence-aware inter-modal conflict resolver."""

    # ANSI colours for terminal
    GREEN  = "\033[92m"
    BLUE   = "\033[94m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"

    def __init__(self, tau: float = 0.20):
        """
        tau : float  — confidence-differential threshold (0–1).
              Default 0.20 (paper recommendation).
        """
        assert 0 < tau < 1, "tau must be between 0 and 1"
        self.tau = tau

    def resolve(self, Ef: str, Cf: float,
                Es: str, Cs: float) -> dict:
        """
        Run EDR algorithm.

        Parameters
        ----------
        Ef : facial emotion label  (str)
        Cf : facial confidence     (float 0-1)
        Es : speech emotion label  (str)
        Cs : speech confidence     (float 0-1)

        Returns dict with keys:
            final_emotion, decision, delta, reasoning, dominant_src
        """
        Cf, Cs = float(Cf), float(Cs)
        delta  = round(abs(Cf - Cs), 4)

        # ── Step 1 : Agreement ────────────────────────────────────
        if Ef.lower() == Es.lower():
            return dict(
                final_emotion = Ef,
                decision      = "AGREEMENT",
                delta         = delta,
                reasoning     = f"Both agree on '{Ef}' (Cf={Cf:.2f}, Cs={Cs:.2f})",
                dominant_src  = None,
            )

        # ── Step 2 & 3 : Dominance ───────────────────────────────
        if delta > self.tau:
            if Cf >= Cs:
                return dict(
                    final_emotion = Ef,
                    decision      = "DOMINANT_FACE",
                    delta         = delta,
                    reasoning     = (f"Face='{Ef}'({Cf:.2f}) vs Speech='{Es}'({Cs:.2f}) "
                                     f"| Δ={delta:.3f}>τ={self.tau} → Face wins"),
                    dominant_src  = "Facial CNN",
                )
            else:
                return dict(
                    final_emotion = Es,
                    decision      = "DOMINANT_SPEECH",
                    delta         = delta,
                    reasoning     = (f"Face='{Ef}'({Cf:.2f}) vs Speech='{Es}'({Cs:.2f}) "
                                     f"| Δ={delta:.3f}>τ={self.tau} → Speech wins"),
                    dominant_src  = "Speech LSTM",
                )

        # ── Step 4 : Uncertain ────────────────────────────────────
        return dict(
            final_emotion = "Uncertain",
            decision      = "UNCERTAIN",
            delta         = delta,
            reasoning     = (f"Face='{Ef}'({Cf:.2f}) vs Speech='{Es}'({Cs:.2f}) "
                             f"| Δ={delta:.3f}≤τ={self.tau} → Mixed/Uncertain"),
            dominant_src  = None,
        )

    def print_result(self, r: dict):
        """Pretty-print result to terminal."""
        colours = {
            "AGREEMENT":       self.GREEN,
            "DOMINANT_FACE":   self.BLUE,
            "DOMINANT_SPEECH": self.BLUE,
            "UNCERTAIN":       self.YELLOW,
        }
        c = colours.get(r["decision"], "")
        print(f"\n{'─'*52}")
        print(f"  {self.BOLD}EDR DECISION : "
              f"{c}{r['decision']}{self.RESET}")
        print(f"  Final Emotion: {c}{self.BOLD}"
              f"{r['final_emotion']}{self.RESET}")
        print(f"  Δ = {r['delta']:.4f}   τ = {self.tau}")
        if r["dominant_src"]:
            print(f"  Winner       : {r['dominant_src']}")
        print(f"  Reason       : {r['reasoning']}")
        print(f"{'─'*52}\n")
