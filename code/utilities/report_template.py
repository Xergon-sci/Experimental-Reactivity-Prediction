import os
from datetime import date
from fpdf import FPDF

class Template(FPDF):

    def __init__(self, title, subtitle, author, modelname):
        super().__init__()
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.IMG = os.path.join(self.PATH, os.pardir, os.pardir, 'reports', 'img')
        self.FONT = os.path.join(self.PATH, os.pardir, os.pardir, 'reports', 'font')
        self.add_font('verdana','', os.path.join(self.FONT, 'verdana.ttf'), uni=True)
        self.add_font('verdana','B', os.path.join(self.FONT, 'verdanab.ttf'), uni=True)
        self.add_font('verdana','I', os.path.join(self.FONT, 'verdanai.ttf'), uni=True)
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.modelname = modelname
        self.today = date.today()
        self.date = self.today.strftime("%d/%m/%Y")
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.frontPage()

    def header(self):
        if self.page_no() != 1:
            self.image(os.path.join(self.IMG, 'vub_driehoek.jpg'), 193, 20, w=10)
            self.set_y(18)

    def footer(self):
        if self.page_no() != 1:
            self.set_text_color(0,0,0)
            self.set_y(-15)
            self.set_font('verdana', size=7)
            self.cell(self.get_string_width(self.modelname), 3, txt=self.modelname, ln=1)
            self.cell(50, 3, txt='© {} {}'.format(self.today.year, self.author), ln=0)
            self.set_x(-30)
            self.set_font('verdana', size=10)
            self.cell(0, 10, str(self.page_no()),'C', ln=2)
    
    def frontPage(self):
        self.add_page()
        self.image(os.path.join(self.IMG, 'vub_logo_digitaal.jpg'), w=67.35, h=30)
        self.image(os.path.join(self.IMG, 'vub_driehoek.jpg'), x=150, y=85, w=53, h=150)
        self.set_x(12)
        self.set_font('verdana', size=12)
        self.cell(0, 45, ln=2)
        self.set_font('verdana', size=27)
        self.set_text_color(0,51,153)
        self.multi_cell(100, 10, txt=self.title, ln=2)
        self.set_font('verdana', size=14)
        self.set_text_color(255,102,0)
        self.multi_cell(100, 10, txt=self.subtitle, ln=2)
        self.cell(0, 150, ln=2)
        self.set_font('verdana', size=12)
        self.set_text_color(0,0,0)
        self.set_y(-40)
        self.multi_cell(50, 5, txt='{}\n{}'.format(self.author, self.date), ln=2)
    
    def head1(self, text):
        self.set_font('verdana', size=16)
        self.set_text_color(0,51,153)
        self.cell(self.get_string_width(text), 10, txt=text, ln=1)
    
    def head2(self, text):
        self.set_font('verdana', size=11)
        self.set_text_color(0,0,0)
        self.cell(self.get_string_width(text), 10, txt=text, ln=1)
    
    def text(self, text):
        self.set_text_color(0,0,0)
        self.set_font('verdana', size=9)
        self.multi_cell(0, 5, txt=text, ln=1)
    
    def log(self, text):
        self.set_text_color(0,0,0)
        self.set_font('verdana', size=7)
        self.multi_cell(0, 5, txt=text, ln=1)
    
    def label(self, label, text):
        text = str(text)
        self.set_text_color(0,0,0)
        self.set_font('verdana', size=9, style='B')
        self.cell(self.get_string_width(label),5, label,ln=0)
        self.set_font('verdana', size=9, style='')
        self.cell(self.get_string_width(text),5, text, ln=1)
    
    def dict_label(self, label, text):
        label = '{} :'.format(label)
        text = str(text)
        self.set_text_color(0,0,0)
        self.set_font('verdana', size=9, style='B')
        self.cell(self.get_string_width(label),5, label,ln=0)
        self.set_font('verdana', size=9, style='')
        self.cell(self.get_string_width(text),5, text, ln=1)