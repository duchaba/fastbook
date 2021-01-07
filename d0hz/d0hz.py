# imports
import graphviz
import gc
import psutil
import torch
import pynvml
import fastai
import fastai.vision
import fastai.text
import fastai.text.models
import fastai.text.data
import fastai.text.learner
import pathlib
import torchvision
import matplotlib
import PIL
import numpy
import pathlib
#
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
  #
  #
  # draw journey
  # @add_method(D0HZ)
  def draw_journey(self):
    x = '"A Rock\n(Start)"[shape=house style=filled fillcolor="' + self.color_yellow + '" ] '
    x += '"A Lake"[shape=box style=filled fillcolor="' + self.color_teal + '" ] '
    #
    x += '"A Rock\n(Start)"->Rocky[label="Learn intro.\nconcept, coding" fontsize=10.0];'
    x += 'Rocky->"A Lake"[label="Make it\nhis journey" fontsize=10.0];'
    #
    self._draw_graph_viz(x,"The Journey",default_shape="circle",fill_color=self.color_pink)
    return
  #
  #
  # draw history fastbook.gv()
  # @add_method(D0HZ)
  def draw_brief_history(self):
    x = '"Artificial\nNueron"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += 'Mark_1[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += 'Perceptron[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Multiple\nLayers"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += 'PDP[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Today Deep\nLearning"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Artificial\nNueron"->Mark_1->Perceptron->"Multiple\nLayers"->PDP->"Today Deep\nLearning";'
    #
    x += '1943->"Artificial\nNueron";"Warren\nMcCulloch"->"Artificial\nNueron";"Walter\nPitts"->"Artificial\nNueron";'
    x += '"Frank\nRosenblatt"->Mark_1;Machine->Mark_1;'
    x += '"Marvin\nMinsky"->Perceptron;"Seymoun\nPapert"->Perceptron;'
    x += 'Perceptron->"Single\nLayer";Perceptron->"Two\nLayers";'
    x += '"Single\nLayer"->Failed;"Two\nLayers"->"Minimum\nSuccess";'
    x += '"David\nRumelheart"->PDP;"James\nMcClellan"->PDP;1986->PDP;'
    x += '2020->"Today Deep\nLearning";"fast.ai"->"Today Deep\nLearning";'
    #
    self._draw_graph_viz(x,"Brief History",default_shape="component",direction="TB",graph_size="8,8!")
    return
  #
  #
  # draw fastai team
  # @add_method(D0HZ)
  def draw_fastai_team(self):
    x = '"Fast.ai"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Jeremy\nHoward"[shape=Msquare style=filled fillcolor="' + self.color_teal + '" ] '
    x += '"Dr. Rachel\nThomas"[shape=Msquare style=filled fillcolor="' + self.color_cyan + '" ] '
    x += '"Sylvain\nGugger"[shape=Msquare style=filled fillcolor="' + self.color_cyan + '" ] '
    x += '"Alexis\nGallagher"[shape=Msquare style=filled fillcolor="' + self.color_cyan + '" ] '
    x += '"Rocky\n:-)"[shape=circle style=filled fillcolor="' + self.color_yellow + '" ] '
    #
    x += '"Jeremy\nHoward"->"Fast.ai"->"Sylvain\nGugger"->"Thousands of\nStudents\n& AI Community"->"Duc\nHaba"->"Rocky\n:-)";'
    x += '"Fast.ai"->"Dr. Rachel\nThomas";"Fast.ai"->"Alexis\nGallagher";'
    #
    self._draw_graph_viz(x,"Fast.ai Team to Rocky",default_shape="component",arrow_head="dot")
    return
  #
  #
  # draw learing process
  # @add_method(D0HZ)
  def draw_learning_process(self):
    x = '"Do It\n(Code)"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Code It"[shape=circle style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Share\nIt"[shape=circle style=filled fillcolor="' + self.color_yellow + '" ] '
    x += '"Start\nYour Project"[shape=component style=filled fillcolor="' + self.color_teal + '" ] '
    #
    x += '"Learn\nBasic Concept"->"Do It\n(Code)"->"Better\nUnderstanding"->"Deeper Concept";'
    x += '"Deeper Concept"->"Do It\n(Code)"[label="practice"];'
    x += '"Start\nYour Project"->"Code It"->"Test It"->"Refine\nProject";'
    x += '"Refine\nProject"->"Code It"[label="coding"];'
    x += '"Do It\n(Code)"->"Start\nYour Project";'
    x += '"Test It"->"Share\nIt";'
    #
    self._draw_graph_viz(x,"Deep Learning Process",default_shape="component", direction="TB")
    return
  #
  #
  # draw dev stack
  # @add_method(D0HZ)
  def draw_dev_stack(self):
    x = '"Python\n3.6+"->PyTorch->"Fast.ai"->"Jupyter\nNotebook"'
    #
    self._draw_graph_viz(x,"Deep Learning Dev Stack",default_shape="Mcircle", fill_color=self.color_orange)
    return
  #
  #
  # draw dev stack
  # @add_method(D0HZ)
  def print_ram_info(self):
    gpu_total = gpu_free = cpu_free = gc_free = 0
    self._ph()
    try:
      gc_free = gc.collect()
      torch.cuda.empty_cache()  # @UndefinedVariable
      val = psutil.virtual_memory()._asdict()
      cpu_free = round((val["available"] / (1024**3)), 2)
      self._pp("Free CPU RAM", str(cpu_free) + " GB")
      #
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      gpu_free = round(info.free / (1024**3), 2)
      self._pp("Free GPU RAM", str(gpu_free) + " GB")
      #
      gpu_total = round(info.total / (1024**3), 2)
      self._pp("Total GPU RAM", str(gpu_free) + " GB")
      self._pp("Garbage Collection", gc_free)
    except:
      self._pp("**Error", "NO GPU accelerator")
      self._pp("Suggest recovery", "Menu > Runtime > Change Runtime Type > {select} GPU accelerator")
    self._ph()
    return
  #
  #
  # draw diff between fast.ai version 1 and version 2
  # @add_method(D0HZ)
  def draw_fastai_v1_v2(self):
    x = '"Image Data\nLoader (V2)"[shape=circle style=filled fillcolor="' + self.color_orange + '" ] '
    x += '"Fine Tune\n(V2)"[shape=circle style=filled fillcolor="' + self.color_orange + '" ] '
    # x += 'edge[splines=curved] '
    #
    x += '"Fetch Image\n(V1)"->Inspect->Clean->"Hyper-parameters"->Load->Split->Label->Augmentation->Databunch->Normalization->Fit_rate->Train->Unfreeze->Fit_rate2->Train2->Review->Inference; '
    x += '"Fetch Image\n(V1)"->"Image Data\nLoader (V2)"->"Fine Tune\n(V2)";'
    x += 'Load->"Image Data\nLoader (V2)"[arrowhead="dot"];'
    x += 'Split->"Image Data\nLoader (V2)"[arrowhead="dot"];'
    x += 'Label->"Image Data\nLoader (V2)"[arrowhead="dot"];'
    x += 'Augmentation->"Image Data\nLoader (V2)"[arrowhead="dot"];'
    x += 'Databunch->"Image Data\nLoader (V2)"[arrowhead="dot"];'
    x += 'Train->"Fine Tune\n(V2)"[arrowhead="dot"];'
    x += 'Unfreeze->"Fine Tune\n(V2)"[arrowhead="dot"];'
    x += 'Train2->"Fine Tune\n(V2)"[arrowhead="dot"];'
    x += 'Fit_rate->"Fine Tune\n(V2)"[arrowhead="dot"];'
    x += 'Fit_rate2->"Fine Tune\n(V2)"[arrowhead="dot"];'
    x += '"Fine Tune\n(V2)"->Inference;'
    x += '"Based Arch,\nTransf., WD, etc."->"Hyper-parameters"[arrowhead="dot"]'
    #
    self._draw_graph_viz(x,"Fast.ai V1 & V2",default_shape="component",direction="TB",graph_size="8,8!")
    return
  #
  #
  #
  # is cat method
  # @add_method(D0HZ)
  def is_a_cat(self,x):
    return x[0].isupper()
  #
  # do the dance
  # @add_method(D0HZ)
  def dance_two_steps_img_id(self, valid_perc=0.2, base_arch=torchvision.models.resnet.resnet34,
    epoch=3, image_size=224):
    #
    # reset them
    self.learn = None
    self.data_loader = None
    self.path = pathlib.Path("/root/.fastai/data/oxford-iiit-pet/images")
    #
    # header
    self._ph()
    self._pp("Dance the Two Steps", "CNN Image Classifiction for Dogs & Cats")
    #
    # clear out RAM with garbage collection
    self.print_ram_info()
    #
    # re-code using the "river" style
    self.data_loader = fastai.vision.data.ImageDataLoaders.from_name_func(
      self.path,
      fastai.data.transforms.get_image_files(self.path),
      valid_pct=valid_perc,
      seed=42,
      label_func=self.is_a_cat,
      # bs=16,
      item_tfms=fastai.vision.augment.Resize(image_size))
    #
    self.learn = fastai.vision.learner.cnn_learner(self.data_loader, base_arch, metrics=fastai.metrics.error_rate)
    self.learn.fine_tune(epoch)
    self._ph()
    self._pp("Display", "Mini-batch (data-bunch old term.)")
    self.data_loader.show_batch()
    self._ph()
    self._pp("Display", "Learn results")
    self.learn.show_results()
    return
  #
  #
  # say is dog or cat
  # @add_method(D0HZ)
  def say_pet_is(self,img_path):
    img = fastai.vision.core.PILImage.create(img_path)
    x,y,z = self.learn.predict(img)
    dogcat = numpy.array(z)
    if (dogcat[0] > dogcat[1]):
      ans = "dog"
    else:
      ans = "cat"
    self._ph()
    self._pp("Rocky answer is", ans)
    self._pp("Confidence probability is", str(round(dogcat.max() * 100)) + "%")
    self._ph()
    #
    display(img)
    return
  #
  #
  # draw ann (artificial neural network)
  # @add_method(D0HZ)
  def draw_ann(self):
    x = '"Architecture\n(Math Func.)"[shape=circle style=filled fillcolor="' + self.color_pink + '" ]; '
    x += '"Model\n(current)"[shape=circle style=filled fillcolor="' + self.color_pink + '" ]; '
    x += '"Pretrain Model\n(resnet, etc.)"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_green + '" ]; '
    x += '"universal\napproximation\ntheorem"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Train\ndata set"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Valid\ndata set"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Test\ndata set"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Data\nBias"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Data\nAugmentation"[shape=box style=rounded fontsize=10.0 fillcolor="' + self.color_gray20 + '" ]; '
    x += '"Deploy\nModel"[shape=circle fontsize=11.0 fontcolor="#f0f0f0" fillcolor="' + self.color_indigo + '" ]; '
    #
    x += '"Architecture\n(Math Func.)"->"Prediction\n(results)"->"Loss"; '
    x += '"Parameters\n(weights)"->"Architecture\n(Math Func.)"[label="uase positive\nfeedback loop" fontsize=8.0]; '
    x += '"Loss"->"Parameters\n(weights)"[constraint=false label="update loop (Train or Fit)\n(SGD minimize loss)\nEpoch (1 complete cycle)", fontsize=8.0]; '
    x += 'Label->"Loss"[label="predict what", fontsize=8.0]; '
    x += '"universal\napproximation\ntheorem"->"Architecture\n(Math Func.)"[label="use as\nbase concept" arrowhead="dot" fontsize=8.0 style=dashed]; '
    x += '"Train\ndata set"->"Architecture\n(Math Func.)"[label="learn" arrowhead="dot" fontsize=8.0 style=dashed]; '
    x += '"Valid\ndata set"->"Loss"[label="learn" arrowhead="dot" fontsize=8.0 style=dashed]; '
    x += '"Architecture\n(Math Func.)"->Label[label="learn" arrowhead="dot" fontsize=8.0 style=dashed]; '
    x += '"Input"->"Architecture\n(Math Func.)"[label="use Data-Loader\n(Mini-batch)" fontsize=8.0]; '
    #
    x += '"Architecture\n(Math Func.)"->"Model\n(current)"[label="instantiage new",fontsize=8.0, style="dashed", arrowhead="dot"]; '
    x += '"Pretrain Model\n(resnet, etc.)"->"Architecture\n(Math Func.)"[label="use as\nbased model\n(Fine-tune)" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    x += '"Metric"->Loss[label="specify loss\nfunction" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    x += '"Architecture\n(Math Func.)"->"Metric"[label="measure accuracy" arrowhead="dot" fontsize=8.0 style="dashed"]; '
    #
    x += 'Loss->"Deploy\nModel"[fontsize=8.0 label="export after\ntraining completed" style="dashed"]; '
    x += '"Test\ndata set"->"Deploy\nModel"[label="validate accuracy\n(avoid overfitting)" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    x += '"Data\nAugmentation"->"Architecture\n(Math Func.)"[label="increase input\ndata" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    x += '"Data\nBias"->"Deploy\nModel"[label="test real-world data\n(out of domain data)" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    x += '"Deploy\nModel"->"Loss"[constraint=false label="Incremental Update\n(avoid negative feedback loop)" fontsize=8.0 style="dashed" arrowhead="dot"]; '
    #
    self._draw_graph_viz(x,"Machine Learning & Artificial Neural Network (ANN) & CNN\nAuthor Samul 1949 & Fast.ai 2.0",default_shape="component", fill_color=self.color_yellow)
    return
  #
  #
  # draw fastai arch (high)
  # @add_method(D0HZ)
  def draw_fastai_arch_high(self):
    x = '"High Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Mid Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Low Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    #
    x += '"Layer Architecture"[shape=invhouse style=filled fillcolor="' + self.color_teal + '" ] '
    x += '"Vision"[shape=Mcircle style=filled fillcolor="' + self.color_yellow + '" ] '
    x += '"Text"[shape=Mcircle style=filled fillcolor="' + self.color_yellow + '" ] '
    x += '"Tabular"[shape=Mcircle style=filled fillcolor="' + self.color_yellow + '" ] '
    x += '"Collab.\nFiltering"[shape=Mcircle style=filled fillcolor="' + self.color_yellow + '" ] '
    # high
    x += '"High Level\nLayer API"->"Mid Level\nLayer API"->"Low Level\nLayer API"[arrowhead="invodot"]'
    x += '"High Level\nLayer API"->"Vision"[constraint=false]'
    x += '"High Level\nLayer API"->"Text"[constraint=false]'
    x += '"High Level\nLayer API"->"Tabular"[constraint=false]'
    x += '"High Level\nLayer API"->"Collab.\nFiltering"[constraint=false]'
    #
    x += '"Vision"->"Layer Architecture"[style="dashed" arrowhead="dot" fontsize=8.0 label="consistency\naccross domains\n(Classification)"]'
    x += '"Text"->"Layer Architecture"[style="dashed" arrowhead="dot" fontsize=8.0 label="(Classification)"]'
    x += '"Tabular"->"Layer Architecture"[style="dashed" arrowhead="dot" fontsize=8.0 label="(Classification &\nRegression)"]'
    x += '"Collab.\nFiltering"->"Layer Architecture"[style="dashed" arrowhead="dot" fontsize=8.0 label="(Regression)"]'
    #
    x += '"Layer Architecture"->".show_batch()"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->".fit_one_cycle()"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->".show_results()"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Data Block\nClass"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Learner\nClass"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Incrementally\nAdapting"[arrowhead="dot" style="dashed"]'
    #
    self._draw_graph_viz(x,"Fast.ai Architecture Version 2.0\nHigh Level",default_shape="component",
      direction="TD", node_font_size=9.0)
    return
  #
  #
  # draw fastai arch (mid)
  # @add_method(D0HZ)
  def draw_fastai_arch_mid(self):
    x = '"High Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Mid Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Low Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    #
    x += '"Layer Architecture"[shape=invhouse style=filled fillcolor="' + self.color_teal + '" ] '
    x += '"Two-way\nCallbacks"[fillcolor="' + self.color_blue + '" fontcolor="#f0f0f0" ] '
    # mid
    x += '"High Level\nLayer API"->"Mid Level\nLayer API"->"Low Level\nLayer API"[arrowhead="invodot"]'
    x += '"Mid Level\nLayer API"->"Layer Architecture"[constraint=false]'
    #
    x += '"Layer Architecture"->"Learner\nClass"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Two-way\nCallbacks"[arrowhead="dot" style="dashed" label="Rocky\nFavorite" fontsize=9.0]'
    x += '"Layer Architecture"->Optimizer[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Metrics API"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Data\nExtern."[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Data\nLoader"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Data\nCore"[arrowhead="dot" style="dashed"]'
    # x += '"Two-way\nCallbacks"->'
    #
    self._draw_graph_viz(x,"Fast.ai Architecture Version 2.0\nMid Level",default_shape="component",
      direction="TD", node_font_size=9.0)
    return
  #
  #
  # draw fastai arch (low)
  # @add_method(D0HZ)
  def draw_fastai_arch_low(self):
    x = '"High Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Mid Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    x += '"Low Level\nLayer API"[shape=circle fontsize=11.0 style=filled fillcolor="' + self.color_pink + '" ] '
    #
    x += '"Layer Architecture"[shape=invhouse style=filled fillcolor="' + self.color_teal + '" ] '
    # mid
    x += '"High Level\nLayer API"->"Mid Level\nLayer API"->"Low Level\nLayer API"[arrowhead="invodot"]'
    x += '"Low Level\nLayer API"->"Layer Architecture"[constraint=false]'
    #
    x += '"Layer Architecture"->"PyTorch\nFoundation"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Transforms"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Piplines"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Dispatch"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Semantic\nTensors"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"GPU-accelerated"[arrowhead="dot" style="dashed"]'
    x += '"Layer Architecture"->"Convience Fun."[arrowhead="dot" style="dashed"]'
    #
    self._draw_graph_viz(x,"Fast.ai Architecture Version 2.0\nLow Level",default_shape="component",
      direction="TD", node_font_size=9.0)
    return
  #
  #
  # do the dance dosido
  # @add_method(D0HZ)
  def dance_dosido_img_segm(self, base_arch=torchvision.models.resnet.resnet34,
    epoch=8):
    #
    # reset them
    self.learn = None
    self.data_loader = None
    self.path = pathlib.Path("/root/.fastai/data/camvid_tiny")
    #
    # header
    self._ph()
    self._pp("Dance the Dosido", "Street, Car Segmentation from Camvid")
    #
    # clear out RAM with garbage collection
    self.print_ram_info()
    #
    self.data_loader = fastai.vision.data.SegmentationDataLoaders.from_label_func(
      self.path,
      bs = 8,
      fnames = fastai.data.transforms.get_image_files(self.path/"images"),
      label_func = lambda o: self.path/'labels'/f'{o.stem}_P{o.suffix}',
      codes = numpy.loadtxt(self.path/'codes.txt', dtype=str))
    #
    self.learn = fastai.vision.learner.unet_learner(self.data_loader, base_arch)
    self.learn.fine_tune(epoch)
    #
    self._ph()
    self._pp("Display", "Mini-batch (data-bunch old term.) & Results")
    self._ph()
    self.data_loader.show_batch()
    self.learn.show_results(max_n=6, figsize=(8,8))
    return
  #
  #
  # do the dance
  # @add_method(D0HZ)
  def dance_monster_mash_nlp(self, base_arch=fastai.text.models.awdlstm.AWD_LSTM,
      epoch=6, drop_out=0.5, fit_rate=0.01):
    #
    # reset them
    self.learn = None
    self.data_loader = None
    self.path = pathlib.Path("/root/.fastai/data/imdb")
    #
    # header
    self._ph()
    self._pp("Dance the Monster Mash", "IMDB movie review NLP")
    #
    # clear out RAM with garbage collection
    self.print_ram_info()
    #
    self.data_loader = fastai.text.data.TextDataLoaders.from_folder(
      self.path,
      valid='test')
    #
    self.learn = fastai.text.learner.text_classifier_learner(self.data_loader, base_arch,
      drop_mult=drop_out,
      metrics=fastai.metrics.accuracy)
    self.learn.fine_tune(epoch,fit_rate)
    #
    self._ph()
    self._pp("Display", "Mini-batch (data-bunch old term.) & Results")
    self._ph()
    self.data_loader.show_batch()
    self.learn.show_results()
    return
  #
  #
  # say sentiment is...
  # @add_method(D0HZ)
  def say_sentiment_is(self,msg):
    x,y,z = self.learn.predict(msg)
    senti = numpy.array(z)
    if (senti[0] > senti[1]):
      ans = "Negative sentiment"
    else:
      ans = "Positive sentiment"
    self._ph()
    self._pp("Rocky answer is", ans)
    self._pp("Confidence probability is", str(round(senti.max() * 100)) + "%")
    self._pp("Input", msg)
    self._ph()
    #
    return
  #
  #
  # do the dance
  # @add_method(D0HZ)
  def dance_twosteps_tabular(self, epoch=6, data_path="/root/.fastai/data/adult_sample"):
    #
    # reset them
    self.learn = None
    self.data_loader = None
    self.path = pathlib.Path(data_path)
    self.categorical = ['workclass', 'education', 'marital-status', 'occupation',
      'relationship', 'race']
    self.continuous = ['age', 'fnlwgt', 'education-num']
    self.target = "salary"
    self.train_process = [fastai.tabular.core.Categorify, fastai.tabular.core.FillMissing,
      fastai.data.transforms.Normalize]
    #
    # header
    self._ph()
    self._pp("Dance the Texas Two Steps", "Tablular data from adults data set.")
    #
    # clear out RAM with garbage collection
    self.print_ram_info()
    #
    # data loader
    self.data_loader = fastai.tabular.data.TabularDataLoaders.from_csv(
      self.path/'adult.csv',
      path=self.path,
      y_names=self.target,
      cat_names = self.categorical,
      cont_names = self.continuous,
      procs = self.train_process
      )
    #
    # No based-architecture
    self.learn = fastai.tabular.learner.tabular_learner(self.data_loader,
      metrics=fastai.metrics.accuracy)
    #
    self.learn.fit_one_cycle(epoch)
    #
    self._ph()
    self._pp("Display", "Mini-batch (data-bunch old term.) & Results")
    self._ph()
    self.data_loader.show_batch()
    self.learn.show_results()
    return
  #
  #
  #
  # do the dance
  # @add_method(D0HZ)
  def dance_hula_colab(self, epoch=6, data_path="/root/.fastai/data/movie_lens_sample", y_target_range=(0.5,5.5)):
    #
    # reset them
    self.learn = None
    self.data_loader = None
    self.path = pathlib.Path(data_path)
    #
    # header
    self._ph()
    self._pp("Dance the Hula", "Collab Movie Review rating")
    #
    # clear out RAM with garbage collection
    self.print_ram_info()
    #
    # data loader
    self.data_loader = fastai.collab.CollabDataLoaders.from_csv(self.path/'ratings.csv')
    # learner
    self.learn = fastai.collab.collab_learner(self.data_loader, y_range=y_target_range)
    #
    self.learn.fine_tune(epoch)
    #
    self._ph()
    self._pp("Display", "Mini-batch (data-bunch old term.) & Results")
    self._ph()
    self.data_loader.show_batch()
    self.learn.show_results()
    return
  #
  #
# ---end of D0HZ class
