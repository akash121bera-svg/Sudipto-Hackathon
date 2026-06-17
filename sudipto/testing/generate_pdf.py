import json
import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Draw top accent bar
        self.set_fill_color(43, 108, 176) # Cool Blue
        self.rect(0, 0, 210, 4, 'F')
        
        # Helvetica bold
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(26, 54, 93) # Deep Blue
        self.cell(0, 10, 'AI Youth & Employment Scheme Navigator', 0, 1, 'C')
        
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(113, 128, 150) # Gray
        self.cell(0, 5, 'Comprehensive Test Queries & Expected Eligibility Matches (2026)', 0, 1, 'C')
        self.ln(8)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(160, 174, 192)
        # Draw a thin footer line
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        # Page numbers
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_pdf():
    # Read queries file
    queries_path = os.path.join(os.path.dirname(__file__), 'queries.json')
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    for tc in queries:
        # Check if page break is needed to prevent card splitting
        if pdf.get_y() > 210:
            pdf.add_page()
            
        # Test Case ID & Title Header
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(43, 108, 176) # Theme Primary Blue
        pdf.cell(0, 7, f"{tc['id']}: {tc['description']}", 0, 1)
        
        # User Query Text Box (italic, light background)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(45, 55, 72)
        pdf.set_fill_color(247, 250, 252) # Very light gray
        pdf.set_draw_color(226, 232, 240) # Border grey
        
        query_text = f"Query: \"{tc['query']}\""
        pdf.multi_cell(0, 6, query_text, border=1, fill=True)
        pdf.ln(3)
        
        # Section Titles
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(74, 85, 104)
        pdf.cell(90, 6, "Expected Profile Extraction:", 0, 0)
        pdf.cell(0, 6, "Expected Scheme Matches:", 0, 1)
        
        # Build Profile string content
        profile = tc['expected_profile']
        profile_lines = []
        for k, v in profile.items():
            disp_val = v if v is not None else "-"
            if k == 'annual_income' and isinstance(v, int):
                disp_val = f"Rs. {v:,}"
            profile_lines.append(f"  - {k.replace('_', ' ').capitalize()}: {disp_val}")
        profile_str = "\n".join(profile_lines)
        
        # Build Expected matches content
        matches_str = "\n".join([f"  - {m}" for m in tc['expected_matches']])
        
        # Print columns side-by-side
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(45, 55, 72)
        y_start = pdf.get_y()
        
        # Left Column: Profile Attributes
        pdf.set_x(10)
        pdf.multi_cell(90, 5, profile_str, border=0)
        y_profile_end = pdf.get_y()
        
        # Right Column: Matched Schemes
        pdf.set_y(y_start)
        pdf.set_x(105)
        pdf.multi_cell(95, 5, matches_str, border=0)
        y_matches_end = pdf.get_y()
        
        # Move pointer to bottom of section plus vertical spacing
        max_y = max(y_profile_end, y_matches_end)
        pdf.set_y(max_y)
        
        # Divider line between test cases
        pdf.set_draw_color(242, 244, 248)
        pdf.line(10, pdf.get_y() + 4, 200, pdf.get_y() + 4)
        pdf.ln(8)
        
    pdf_output_path = os.path.join(os.path.dirname(__file__), 'test_queries.pdf')
    pdf.output(pdf_output_path)
    print(f"PDF generated successfully at {pdf_output_path}")

if __name__ == '__main__':
    generate_pdf()
