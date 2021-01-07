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
# ---end of D0HZ class
