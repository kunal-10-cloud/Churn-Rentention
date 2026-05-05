import datetime
from fpdf import FPDF
from typing import Dict, Any

class RetentionReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.set_text_color(0, 51, 102) # Dark blue
        self.cell(0, 10, "ChurnGuard AI - Retention Strategy Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def _safe(s):
    if not isinstance(s, str):
        s = str(s)
    # Replace common unicode punctuation with basic ascii
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "*", "🛡️": "", "✅": "", 
        "🤖": "", "⚠️": "", "📉": "", "📈": "", "⚡": ""
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Force to latin-1 to avoid fpdf unicode parsing width calculation errors
    return s.encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf(report: Dict[str, Any], customer_data: Dict[str, Any]) -> bytes:
    """Generate a PDF document retaining the AI advisor strategy."""
    pdf = RetentionReportPDF()
    pdf.add_page()
    
    # Generated Date
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Customer Data Section
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Customer Profile", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_fill_color(240, 240, 240)
    
    col_width = pdf.epw / 2
    for i, (key, value) in enumerate(customer_data.items()):
        fill = bool(i % 2 == 0)
        pdf.cell(col_width, 8, _safe(f"{key}:"), fill=fill)
        pdf.cell(col_width, 8, _safe(str(value)), fill=fill, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(5)
    
    # Risk Summary
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Risk Assessment", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "B", 10)
    risk_level = report.get('risk_level', 'Unknown')
    if risk_level == "High":
        pdf.set_text_color(200, 0, 0)
    elif risk_level == "Medium":
        pdf.set_text_color(200, 100, 0)
    else:
        pdf.set_text_color(0, 150, 0)
        
    pdf.cell(0, 8, _safe(f"Risk Level: {risk_level}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 10)
    
    if "risk_summary" in report:
        pdf.multi_cell(0, 6, _safe(str(report.get("risk_summary"))), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Contributing Factors
    if "contributing_factors" in report and report["contributing_factors"]:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Contributing Factors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        for factor in report["contributing_factors"]:
            pdf.multi_cell(0, 6, _safe(f"- {factor}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    # Recommended Actions
    if "recommended_actions" in report and report["recommended_actions"]:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Recommended Actions", new_x="LMARGIN", new_y="NEXT")
        for idx, act in enumerate(report["recommended_actions"], 1):
            pdf.set_font("helvetica", "B", 10)
            priority = act.get('priority', '')
            pdf.multi_cell(0, 6, _safe(f"{idx}. {act.get('action')} [Priority: {priority}]"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "I", 10)
            pdf.multi_cell(0, 6, _safe(f"   Rationale: {act.get('rationale')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    # Disclaimers
    if "disclaimers" in report and report["disclaimers"]:
        pdf.set_font("helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, "Important Disclaimers", new_x="LMARGIN", new_y="NEXT")
        for disc in report["disclaimers"]:
            pdf.multi_cell(0, 5, _safe(f"* {disc}"), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
