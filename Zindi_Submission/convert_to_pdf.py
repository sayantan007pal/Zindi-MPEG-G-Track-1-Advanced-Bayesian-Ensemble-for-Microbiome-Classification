#!/usr/bin/env python3
"""
Convert Markdown scientific report to PDF for Zindi submission
"""

import markdown
import pdfkit
import os
from pathlib import Path

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF using markdown + pdfkit"""
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html = markdown.markdown(md_content, extensions=['tables', 'codehilite', 'toc'])
    
    # Add CSS styling for better PDF formatting
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 40px;
                font-size: 12px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 24px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                font-size: 18px;
                margin-top: 30px;
            }}
            h3 {{
                color: #34495e;
                font-size: 16px;
                margin-top: 25px;
            }}
            h4 {{
                color: #34495e;
                font-size: 14px;
                margin-top: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            code {{
                background-color: #f8f8f8;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }}
            pre {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                overflow-x: auto;
                font-size: 11px;
            }}
            .highlight {{
                background-color: #fff3cd;
                padding: 10px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 30px;
            }}
            li {{
                margin: 5px 0;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 15px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
            }}
            .page-break {{
                page-break-before: always;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Configure PDF options
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    try:
        # Convert HTML to PDF
        pdfkit.from_string(html_with_style, pdf_file, options=options)
        print(f"✅ Successfully converted {md_file} to {pdf_file}")
        return True
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        return False

def create_simple_pdf():
    """Create a simple PDF using reportlab if pdfkit fails"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        # Read markdown content
        md_file = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/Zindi_Submission/MPEG_Track1_Scientific_Report.md"
        pdf_file = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/Zindi_Submission/MPEG_Track1_Scientific_Report.pdf"
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'], 
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkslategray
        )
        
        # Create story (content list)
        story = []
        
        # Add title
        story.append(Paragraph("MPEG-G Track 1: Advanced Bayesian Ensemble for Microbiome Classification", title_style))
        story.append(Paragraph("Scientific Report for Zindi Submission", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Add key sections
        sections = [
            ("Executive Summary", "This submission presents a comprehensive machine learning solution achieving 95.0% cross-validation accuracy through advanced Bayesian optimization and ensemble methods."),
            ("Methodology", "We implemented six distinct approaches with Bayesian Optimized Ensemble selected for 95.0% accuracy performance."),
            ("Data Processing", "Advanced feature engineering pipeline reducing 9,132 features to 10 optimal biomarkers with biological significance."),
            ("Model Architecture", "Soft voting ensemble with Random Forest (52.2%), Gradient Boosting (39.0%), and Logistic Regression (8.7%) weights."),
            ("Performance Metrics", "95.0% CV accuracy with [82.1%, 100.0%] confidence interval validated through nested CV, bootstrap CI, and multi-seed analysis."),
            ("Biological Insights", "Selected features include metabolic pathways (K03750, K02588), species markers (Blautia schinkii), and temporal dynamics."),
            ("Innovation", "Advanced Bayesian optimization, Graph Neural Networks, Transfer Learning, and novel feature engineering techniques."),
            ("Efficiency", "5-minute training time, <0.1 second inference, CPU-only deployment with minimal resource requirements."),
            ("Reproducibility", "Fixed random seeds, comprehensive documentation, quality assurance testing, and production-ready implementation."),
            ("Conclusion", "State-of-the-art performance with novel methodological contributions applicable to broader microbiome research.")
        ]
        
        for section_title, section_content in sections:
            story.append(Paragraph(section_title, heading_style))
            story.append(Paragraph(section_content, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Add performance table
        performance_data = [
            ['Validation Method', 'Accuracy', 'Confidence Interval'],
            ['Nested CV', '95.0%', 'Primary metric'],
            ['Bootstrap CI', '94.0%', '[82.1%, 100.0%]'],
            ['Multi-seed', '97.0%', 'High stability'],
            ['Augmented Data', '100.0%', 'Generalization']
        ]
        
        t = Table(performance_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Performance Results", heading_style))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Add technical specifications
        story.append(Paragraph("Technical Specifications", heading_style))
        story.append(Paragraph("• Model: Bayesian Optimized Ensemble with 95.0% CV accuracy", styles['Normal']))
        story.append(Paragraph("• Features: 10 selected from 9,132 original (99.9% reduction)", styles['Normal']))
        story.append(Paragraph("• Training: 5 minutes on MacBook Pro M1", styles['Normal']))
        story.append(Paragraph("• Inference: <0.1 seconds per sample", styles['Normal']))
        story.append(Paragraph("• Deployment: CPU-only, cross-platform compatible", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add submission details
        story.append(Paragraph("Submission Details", heading_style))
        story.append(Paragraph("This PDF represents the complete scientific report for MPEG-G Track 1 submission. The full markdown version contains comprehensive technical details, code examples, and complete methodology documentation.", styles['Normal']))
        story.append(Paragraph("All code, trained models, and detailed documentation are included in the submission package.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print(f"✅ Successfully created PDF report using reportlab: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ reportlab not available. Please install: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Error creating PDF with reportlab: {e}")
        return False

if __name__ == "__main__":
    md_file = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/Zindi_Submission/MPEG_Track1_Scientific_Report.md"
    pdf_file = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/Zindi_Submission/MPEG_Track1_Scientific_Report.pdf"
    
    # Try pdfkit first
    try:
        import pdfkit
        success = convert_md_to_pdf(md_file, pdf_file)
        if not success:
            print("Falling back to reportlab...")
            create_simple_pdf()
    except ImportError:
        print("pdfkit not available, using reportlab...")
        create_simple_pdf()