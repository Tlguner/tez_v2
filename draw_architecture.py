# -*- coding: utf-8 -*-
"""
Publication-Grade Architecture Diagram Generator
Generates clean, aesthetic IEEE-style diagram of the AI Agent + MCP Server + Python Backend architecture.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Color Palette (Editorial light-paper design)
    bg_color = '#FAFAFA'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    c_agent_bg = '#EFF6FF'
    c_agent_border = '#3B82F6'
    c_agent_title = '#1D4ED8'
    
    c_mcp_bg = '#FFF7ED'
    c_mcp_border = '#EA580C'
    c_mcp_title = '#C2410C'
    
    c_py_bg = '#F0FDF4'
    c_py_border = '#16A34A'
    c_py_title = '#15803D'
    
    c_box = '#FFFFFF'
    c_box_border = '#CBD5E1'
    c_text = '#0F172A'
    c_subtext = '#475569'
    
    # 1. LAYER 1: AI Agent Layer (Top)
    rect1 = patches.FancyBboxPatch((0.8, 5.8), 12.4, 2.3, boxstyle="round,pad=0.15,rounding_size=0.2",
                                  facecolor=c_agent_bg, edgecolor=c_agent_border, linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.2, 7.7, "LAYER 1: AI AGENT CONTROLLER (Google Gemini Flash)", fontsize=13, fontweight='bold', color=c_agent_title)
    
    # Boxes inside Layer 1
    boxes_l1 = [
        (1.1, 6.0, 3.6, 1.4, "Pipeline Planner", "• Determines execution workflow\n• Context-aware reasoning\n• Task decomposition"),
        (5.1, 6.0, 3.8, 1.4, "Code Synthesizer", "• Generates modular Python code\n• Prompt-driven template parsing\n• Dynamic parameter injection"),
        (9.3, 6.0, 3.6, 1.4, "Diagnosis & Self-Healing", "• Analyzes stderr tracebacks\n• Generates unified diff patches\n• Enforces 3-tier recovery loop")
    ]
    for x, y, w, h, title, desc in boxes_l1:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor=c_box, edgecolor=c_box_border, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x+0.2, y+h-0.3, title, fontsize=10.5, fontweight='bold', color=c_text)
        ax.text(x+0.2, y+0.2, desc, fontsize=8.5, color=c_subtext, linespacing=1.3)
        
    # 2. LAYER 2: MCP Server Layer (Middle)
    rect2 = patches.FancyBboxPatch((0.8, 3.1), 12.4, 2.3, boxstyle="round,pad=0.15,rounding_size=0.2",
                                  facecolor=c_mcp_bg, edgecolor=c_mcp_border, linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.2, 5.0, "LAYER 2: MODEL CONTEXT PROTOCOL (MCP) SERVER (Node.js / Express)", fontsize=13, fontweight='bold', color=c_mcp_title)
    
    # Boxes inside Layer 2
    boxes_l2 = [
        (1.1, 3.3, 2.8, 1.4, "JSON-RPC Protocol", "• stdio / SSE transport\n• Bi-directional dispatch\n• Request validation"),
        (4.2, 3.3, 3.2, 1.4, "Modular Tool Suite", "• analyze_dataset_tool\n• clean_dataset_tool\n• extract_features_tool"),
        (7.7, 3.3, 2.9, 1.4, "Execution Engine", "• train_models_tool\n• evaluate_and_plot_tool\n• Optuna Bayesian driver"),
        (10.9, 3.3, 2.0, 1.4, "State & Zod", "• Zod schemas\n• run_state.json\n• healing_log")
    ]
    for x, y, w, h, title, desc in boxes_l2:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor=c_box, edgecolor=c_box_border, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x+0.15, y+h-0.3, title, fontsize=10.5, fontweight='bold', color=c_text)
        ax.text(x+0.15, y+0.2, desc, fontsize=8.5, color=c_subtext, linespacing=1.3)
        
    # 3. LAYER 3: Python Execution Engine & Hardware (Bottom)
    rect3 = patches.FancyBboxPatch((0.8, 0.4), 12.4, 2.3, boxstyle="round,pad=0.15,rounding_size=0.2",
                                  facecolor=c_py_bg, edgecolor=c_py_border, linewidth=2)
    ax.add_patch(rect3)
    ax.text(1.2, 2.3, "LAYER 3: COMPUTATIONAL WORKLOAD ENGINE (Python 3.11 / PyTorch CUDA)", fontsize=13, fontweight='bold', color=c_py_title)
    
    # Boxes inside Layer 3
    boxes_l3 = [
        (1.1, 0.6, 3.6, 1.4, "10,004-dim Hybrid Feature Space", "• 10,000 TF-IDF n-grams (1,2)\n• 4 hand-crafted cybersecurity signals\n• Sparse hstack (scipy.sparse)"),
        (5.1, 0.6, 3.8, 1.4, "Deep Learning Neural Topologies", "• Multi-kernel TextCNN (k=3,4,5)\n• BiGRU + 4-Head Self-Attention\n• Sequence Embedding (L=400, d=128)"),
        (9.3, 0.6, 3.6, 1.4, "Hardware Monitoring Subsystem", "• psutil resident memory tracking\n• Wall-clock training latency\n• NVIDIA RTX 5070 CUDA Profiler")
    ]
    for x, y, w, h, title, desc in boxes_l3:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor=c_box, edgecolor=c_box_border, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x+0.2, y+h-0.3, title, fontsize=10.5, fontweight='bold', color=c_text)
        ax.text(x+0.2, y+0.2, desc, fontsize=8.5, color=c_subtext, linespacing=1.3)

    # Inter-layer Connection Arrows
    arrow_props_down = dict(facecolor='#2563EB', edgecolor='none', width=2.5, headwidth=8, headlength=8)
    arrow_props_up = dict(facecolor='#DC2626', edgecolor='none', width=2.5, headwidth=8, headlength=8)
    
    # Layer 1 <-> Layer 2
    ax.annotate("", xy=(3.5, 5.4), xytext=(3.5, 5.8), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=2.5, mutation_scale=15))
    ax.text(3.7, 5.55, "JSON-RPC Tool Invocations", fontsize=8.5, fontweight='bold', color="#1D4ED8")
    
    ax.annotate("", xy=(10.5, 5.8), xytext=(10.5, 5.4), arrowprops=dict(arrowstyle="->", color="#EA580C", lw=2.5, mutation_scale=15))
    ax.text(10.7, 5.55, "Execution Results & Exceptions", fontsize=8.5, fontweight='bold', color="#C2410C")
    
    # Layer 2 <-> Layer 3
    ax.annotate("", xy=(3.5, 2.7), xytext=(3.5, 3.1), arrowprops=dict(arrowstyle="->", color="#EA580C", lw=2.5, mutation_scale=15))
    ax.text(3.7, 2.85, "Subprocess Script Execution", fontsize=8.5, fontweight='bold', color="#C2410C")
    
    ax.annotate("", xy=(10.5, 3.1), xytext=(10.5, 2.7), arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2.5, mutation_scale=15))
    ax.text(10.7, 2.85, "Evaluation Metrics & Artifacts (.npz/.pt)", fontsize=8.5, fontweight='bold', color="#15803D")

    plt.tight_layout()
    
    paths = [
        "architecture_diagram.png",
        "ml-images/architecture_diagram.png",
        "ml-images-82/architecture_diagram.png",
        "ml-images-18/architecture_diagram.png"
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"[SUCCESS] Saved architecture diagram to: {p}")
    plt.close()

if __name__ == "__main__":
    draw_architecture_diagram()
