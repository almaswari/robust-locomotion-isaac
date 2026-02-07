import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # --- STYLES ---
    box_style = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2)
    highlight_style = dict(boxstyle="round,pad=0.3", fc="#e6f3ff", ec="blue", lw=2) # Blue for Standard
    god_mode_style = dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="orange", lw=2) # Orange for God Mode

    # --- BLOCKS ---
    
    # 1. Simulation Layer
    ax.text(6, 7.5, "ENVIRONMENT (ISAAC GYM)", ha="center", fontsize=14, fontweight='bold')
    ax.add_patch(patches.Rectangle((2, 6.5), 8, 1.2, fill=False, linestyle="--", ec="gray"))
    
    ax.text(4, 7, "Robot Physics\n(Joints, Vel)", ha="center", bbox=box_style)
    ax.text(8, 7, "Hidden Physics\n(Friction, Mass, Push)", ha="center", bbox=god_mode_style)

    # 2. Observation Layer (The Split)
    ax.text(6, 5.5, "DATA COLLECTION", ha="center", fontsize=12, fontweight='bold', color='gray')
    
    # Actor Input
    ax.text(4, 4.5, "Actor Obs (48)\n[Blind]", ha="center", bbox=highlight_style)
    
    # Critic Input (The God Mode Connection)
    ax.text(8, 4.5, "Critic Obs (53)\n[Privileged]", ha="center", bbox=god_mode_style)

    # 3. Network Layer
    ax.text(4, 2.5, "ACTOR Network\n(Policy)", ha="center", bbox=highlight_style, fontsize=12)
    ax.text(8, 2.5, "CRITIC Network\n(Value Function)", ha="center", bbox=god_mode_style, fontsize=12)

    # 4. Output Layer
    ax.text(4, 0.5, "Actions\n(Motor Targets)", ha="center", bbox=box_style)
    ax.text(8, 0.5, "Value Estimate\n(Advantage)", ha="center", bbox=box_style)

    # --- ARROWS ---
    # From Sim to Obs
    ax.annotate("", xy=(4, 5), xytext=(4, 6.5), arrowprops=dict(arrowstyle="->", lw=2)) # Physics -> Actor Obs
    ax.annotate("", xy=(8, 5), xytext=(4, 6.5), arrowprops=dict(arrowstyle="->", lw=2)) # Physics -> Critic Obs
    ax.annotate("", xy=(8, 5), xytext=(8, 6.5), arrowprops=dict(arrowstyle="->", lw=2, color="orange")) # Hidden -> Critic Obs

    # From Obs to Networks
    ax.annotate("", xy=(4, 3), xytext=(4, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(8, 3), xytext=(8, 4), arrowprops=dict(arrowstyle="->", lw=2))

    # From Networks to Output
    ax.annotate("", xy=(4, 1), xytext=(4, 2), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(8, 1), xytext=(8, 2), arrowprops=dict(arrowstyle="->", lw=2))

    # Loop back (Action to Sim)
    ax.annotate("", xy=(1.8, 7), xytext=(3.5, 0.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", lw=2, ls="--"))
    ax.text(1, 3.5, "Step(Action)", rotation=90, va="center")

    plt.title("Asymmetric Actor-Critic Architecture (God Mode)", fontsize=16)
    plt.tight_layout()
    plt.savefig("architecture_diagram.png", dpi=300)
    print("Saved architecture_diagram.png")

if __name__ == "__main__":
    draw_architecture()
