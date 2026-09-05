from fpdf import FPDF
import datetime

def generate_maintenance_report(engine_id, rul, status):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AeroGuard: Predictive Maintenance Report", ln=True, align='C')
    
    # Project Info (Your Branding)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Asset ID: Engine Unit #{engine_id}", ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Remaining Useful Life (RUL): {int(rul)} Cycles", ln=True)
    pdf.cell(200, 10, txt=f"Current Health Status: {status}", ln=True)
    
    pdf.ln(5)
    
    # Recommendation Logic
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Maintenance Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    
    if rul < 30:
        recommendation = "URGENT: Schedule immediate inspection. High probability of bearing/blade failure."
    elif rul < 75:
        recommendation = "ADVISORY: Monitor sensor fluctuations. Plan maintenance within the next 15 days."
    else:
        recommendation = "OPTIMAL: No action required. Continue standard operations."
        
    pdf.multi_cell(0, 10, txt=recommendation)
    
    report_name = f"Report_Engine_{engine_id}.pdf"
    pdf.output(report_name)
    return report_name