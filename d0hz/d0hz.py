# create class
class D0HZ(object):
  #
  # initialize the object
  def __init__(self, name="Wallaby"):
    self.author = "Duc Haba"
    self.name = name
    self._ph()
    self._pp("Hello from", self.__class__.__name__)
    self._pp("Code name", self.name)
    self._pp("Author is", self.author)
    self._ph()
    #
    # bootstrap colors
    self.color_blue = "#007bff"
    self.color_indigo = "#6610f2"
    self.color_purple = "#6f42c1"
    self.color_pink = "#e83e8c"
    self.color_red = "#dc3545"
    self.color_orange = "#fd7e14"
    self.color_yellow = "#ffc107"
    self.color_green = "#28a745"
    self.color_teal = "#20c997"
    self.color_cyan = "#17a2b8"
    self.color_gray80 = "#343a40"
    self.color_gray70 = "#495057"
    self.color_gray40 = "#ced4da"
    self.color_gray20 = "#e9ecef"
    return
  #
  # pretty print output name-value line
  def _pp(self, a, b):
    print("%40s : %s" % (str(a), str(b)))
    return
  #
  # pretty print the header or footer lines
  def _ph(self):
    print("-" * 40, ":", "-" * 40)
    return
  #
  # dance
  def dance_happy(self):
    char = "        _=,_\n    o_/6 /#\\\n    \\__ |##/\n     ='|--\\\n       /   #'-.\n"
    char = char + "       \\#|_   _'-. /\n        |/ \\_( # |\" \n       C/ ,--___/\n"
    print(char)
    self._ph()
    return
  #
  # draw black sheep
  def draw_black_sheep(self):
    c = "\n_________________       _       _       _       _       _       _       _ \n"
    c = c + "______________-(_)-  _-(_)-  _-(_)-  _-(\")-  _-(_)-  _-(_)-  _-(_)-  _-(_)- \n"
    c = c + "___________`(___)  `(___)  `(___)  `(###)  `(___)  `(___)  `(___)  `(___) \n"
    c = c + "____________// \\\\   // \\\\   // \\\\   // \\\\   // \\\\   // \\\\   // \\\\   // \\\\ \n"
    print(c)
    self._ph()
    self._pp("bye'aaa", "bye'aaa")
    return
  #
  #
  # draw graph using graphvix.org
  #@add_method(D0HZ)
  def _draw_graph_viz(self,graph,label,direction="LR",arrow_head="normal",bgcolor="#dee2e6",
    edge_color="#6c757d",default_shape="oval",font_color="#343a40", is_filled=True,
    fill_color="#ced4da",label_color="#17a2b8",graph_size="",node_font_size=14.0):
    #
    # set up
    fill = ''
    if (is_filled):
      fill = ' style=filled color="#343a40" fillcolor="' + fill_color + '" '
    gsize =''
    if (graph_size != ""):
      gsize = ' size="' + str(graph_size) + '" '
    x = 'digraph G{'
    x += 'label="' + label + '" '
    x += 'rankdir="' + direction + '" '
    #
    x += 'node [ fontname="Arial" '
    x += fill
    x += 'shape=' + default_shape + ' '
    x += 'fontcolor="' + font_color + '" '
    x += 'fontsize=' + str(node_font_size) + ' '
    x += ' ] '
    #
    x += 'graph [pad=0.4 fontsize=20 fontcolor="' + label_color + '" ' + gsize
    x += 'bgcolor="' + bgcolor + '" '
    x += ' ] '
    #
    x += 'edge ['
    x += 'arrowhead=' + arrow_head + ' '
    x += 'color="' + edge_color + '" '
    x += ' ] '
    #
    x += graph
    x += ' }'
    display(graphviz.Source(x))
    return
  #
  #
  # draw it using fastbook.gv()
  #@add_method(D0HZ)
  def draw_deep_learning_scope(self):
    x = 'Deep_Learning[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += 'Vision->Deep_Learning[label="classify & detect\n(Classification)", fontsize=8.0]; '
    x += '"Text (NLP)"->Deep_Learning[label="classify &\nconverse (weak)\n(Classification)", fontsize=8.0]; '
    x += 'Tabular->Deep_Learning[label="predict\ncardinality data\n(Classification & Regression)", fontsize=8.0]; '
    x += '"Collaborative Filtering\n(Recomendation Sys.)"->Deep_Learning[label="predict &\nrecomend\n(Regression)", fontsize=8.0]; '
    x += 'Deep_Learning->Medicine; '
    x += 'Deep_Learning->Biology; '
    x += 'Deep_Learning->Game_AI; '
    x += 'Deep_Learning->Robotics; '
    x += 'Deep_Learning->"Many Others"; '
    #
    self._draw_graph_viz(x,"Deep Learning Scope",default_shape="component",arrow_head="dot")
    return

# ---end of D0HZ class
