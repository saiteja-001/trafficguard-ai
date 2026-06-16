from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import datetime
from PIL import Image as PILImage

def generate_violation_pdf(original_pil, annotated_pil, junction, vehicle_type, license_plate, violations, confidence_score):
    """
    Generates a professional PDF evidence challan using ReportLab.
    Returns a BytesIO buffer of the generated PDF file.
    """
    buffer = io.BytesIO()
    
    # Create the document template
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        leftMargin=36, 
        rightMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0F172A'), # Slate 900
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E3A8A'), # Navy Blue
        spaceBefore=12,
        spaceAfter=6
    )
    
    text_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'), # Slate 700
        leading=14
    )
    
    badge_style = ParagraphStyle(
        'BadgeText',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#DC2626'), # Red 600
        alignment=1 # Center
    )

    # 1. Header Banner
    header_data = [
        [
            Paragraph("🚦 <b>TRAFFICGUARD AI</b>", ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=18, textColor=colors.white)),
            Paragraph("<b>E-CHALLAN DIGITAL RECORD</b>", ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12, textColor=colors.white, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[270, 270])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E3A8A')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))

    # 2. Case Details Table
    ticket_id = f"TG-{datetime.datetime.now().strftime('%Y%m%d')}-{np_hash(license_plate)}"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metadata_data = [
        ["Ticket Reference ID:", ticket_id, "Date & Time:", date_str],
        ["Camera Junction Location:", junction, "Primary Offender Type:", vehicle_type.upper()],
        ["Extracted License Plate:", f"<b>{license_plate}</b>", "Overall System Confidence:", f"{confidence_score:.1%}"]
    ]
    metadata_table = Table(metadata_data, colWidths=[135, 135, 135, 135])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#475569')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (1,2), (1,2), colors.HexColor('#1E3A8A')), # Bold license plate in blue
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 15))

    # 3. Photographic Evidence (Side-by-Side Images)
    story.append(Paragraph("Photographic Evidence", section_style))
    
    # Resize images for PDF fit (keeping aspect ratio, max width 250 each)
    orig_w, orig_h = original_pil.size
    scale = min(250.0 / orig_w, 180.0 / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    orig_stream = io.BytesIO()
    original_pil.save(orig_stream, format='PNG')
    orig_stream.seek(0)
    pdf_orig_img = Image(orig_stream, width=new_w, height=new_h)
    
    annot_stream = io.BytesIO()
    annotated_pil.save(annot_stream, format='PNG')
    annot_stream.seek(0)
    pdf_annot_img = Image(annot_stream, width=new_w, height=new_h)
    
    img_table_data = [
        [pdf_orig_img, pdf_annot_img],
        [
            Paragraph("<b>Figure 1: Original Capture</b>", ParagraphStyle('Cap1', fontName='Helvetica', fontSize=9, alignment=1, textColor=colors.HexColor('#64748B'))),
            Paragraph("<b>Figure 2: Annotated Violation Analysis</b>", ParagraphStyle('Cap2', fontName='Helvetica', fontSize=9, alignment=1, textColor=colors.HexColor('#64748B')))
        ]
    ]
    img_table = Table(img_table_data, colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    # 4. Violations Summary Table
    story.append(Paragraph("Detected Offenses Details", section_style))
    
    violations_header = [["Violation Type", "Severity", "Explanation", "Confidence"]]
    violations_body = []
    
    for v in violations:
        # v: (type, explanation, severity, confidence)
        v_type, v_exp, v_sev, v_conf = v
        body_row = [
            Paragraph(f"<b>{v_type}</b>", text_style),
            Paragraph(f"<font color='{get_severity_color(v_sev)}'><b>{v_sev}</b></font>", text_style),
            Paragraph(v_exp, text_style),
            Paragraph(f"{v_conf:.1%}", text_style)
        ]
        violations_body.append(body_row)
        
    if not violations_body:
        violations_body.append([
            Paragraph("No Violations", text_style),
            Paragraph("<font color='green'><b>LOW</b></font>", text_style),
            Paragraph("Image complies with traffic rules.", text_style),
            Paragraph("100%", text_style)
        ])
        
    violations_table = Table(violations_header + violations_body, colWidths=[140, 70, 260, 70])
    violations_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(violations_table)
    story.append(Spacer(1, 20))

    # 5. Footer Instructions & Legal Footnote
    footer_text = """
    <b>Instructions for violator:</b> This is an automated enforcement notice issued by TrafficGuard AI on behalf of the Traffic Police Department. Bounding boxes represent photographic proof analyzed by certified computer vision systems. If you wish to appeal this challan, please visit the traffic court website referencing the Ticket ID listed above within 15 days.
    """
    story.append(Paragraph(footer_text, ParagraphStyle('Foot', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#64748B'), leading=11)))
    
    # Build Document
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_severity_color(severity):
    if severity == "HIGH":
        return "#DC2626" # Red
    elif severity == "MEDIUM":
        return "#D97706" # Orange
    elif severity == "INFO":
        return "#2563EB" # Blue
    else:
        return "#16A34A" # Green

def np_hash(text):
    """Produces a short hash code from string for Ticket IDs."""
    if text == "Unreadable" or not text.strip():
        return "XXXX"
    import hashlib
    h = hashlib.sha1(text.strip().encode())
    return h.hexdigest()[:4].upper()
