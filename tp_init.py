from smilPython import *
from smilToNumpyPlot2 import * # from Amin FEHRI
# from save_colors import * # from Beatriz Marcotegui
from smilMorphoPython import *
from smilBasePython import *
from smilCorePython import *
import os
import numpy as np

# Get the absolute path of usefull directories
notebooks_dir = os.path.realpath( os.path.dirname(os.path.realpath(__file__)) + "/.")
images_dir = "/home/commun/tp-morpho/images/"
output_dir = os.path.realpath( notebooks_dir + "/output/")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


Morpho.setDefaultSE(HexSE())

flood = dualBuild
raze = build



se1 =  StrElt(False,(0,1))
se2 =  StrElt(False,(0,2))
se3 =  StrElt(False,(0,3))
se4 =  StrElt(False,(0,4))
se5 =  StrElt(False,(0,5))
se6 =  StrElt(False,(0,6))
se7 =  StrElt(False,(0,7))
se8 =  StrElt(False,(0,8))

seh1 =  StrElt(True,(0,1))
seh2 =  StrElt(True,(0,2))
seh3 =  StrElt(True,(0,3))
seh4 =  StrElt(True,(0,4))
seh5 =  StrElt(True,(0,5))
seh6 =  StrElt(True,(0,6))


def drawBorder(im,val):
    drawRectangle(im,0,0,im.getWidth(),im.getHeight(),val,False)
    return

# ------------------------BEGIN DISP NOTEBOOK ------------------------
# DISPLAYS FOR NOTEBOOK
# display a list of images
def disp(imIn = [], label = []):
    if(type(imIn)!=list):
        imList = [imIn]
        if(type(label)!=list):
            label = [label]
    else:
        imList = imIn
    nb = len(imList)
    plt.figure()
    gs = gridspec.GridSpec(1, nb)
    gs.update(left=0.05, right=1.7, wspace=0.4)
    ax = []
    while len(label) < nb:
        label.append(False)
    for i in range(0, nb):
        ax.append(plt.subplot(gs[0, i]))
        singlePlot(imList[i], ax[i], label[i])
    plt.show()


# interactive display of a single image ('%matplotlib notebook' required)
def dispI(imIn, label=False):
    print("WARNING: FOR INTERACTIVE DISPLAY backend '%matplotlib notebook' IS REQUIRED ") 
    smilToNumpyPlot(imIn,label)
    return

def colorHighlight(im, imBool, color = [0,0,255]):
    '''colorHighlight(im, imBool, color): Highlights a set of pixels of
    the input image im with a color specified by the 3rd argument
    color, supposed to be a list of three channel values [R, G, B] in [0, 255].
    The set of pixels to highlight is determined by the booloean image
    imBool, of the same dimensions as the input image im

    '''
    im3 = combineChannels(im, im, im)
    imCol = Image(im,"RGB")
    chan1, chan2, chan3 = Image(im), Image(im), Image(im)
    chan1 << 0
    chan2 << 0
    chan3 << 0
    add(chan1, color[0], chan1)
    add(chan2, color[1], chan2)
    add(chan3, color[2], chan3)
    imCol = combineChannels(chan1, chan2, chan3)
    compare(imBool, ">", 0, imCol, im3, im3)
    return im3

#binOverlay = colorHighlight
def binOverlay(im, imBool, color = [255,0,0]):
    im2display = colorHighlight(im, imBool, color)
    disp(im2display)
    return im2display
    
# ------------------------END DISP NOTEBOOK ------------------------

def ImDisplay(*args):
    #This function can display an image from morphee in an external viewer.
    #The image if first saved to a temporary file (PNG format) and the viewer
    #is called upon this image. 
    """
    Allows to display images (one or severals) in NXV software
    
    Indicate Morph-M Images Variables : imIn1,imIn2,....	
    Image scalarDataType : sdtUINT8 	
    Options :
    - Indicate the name of image :"image 1","image 2"
    
    NXV source in Qt -- Christophe Clienti CMM-ENSMP-ARMINES 
    ImDisplay Function -- Nicolas Elie CMM-ENSMP-ARMINES 
    """
    #===============================================================================
    # 			A PARAMETRER PAR UTILISATEUR
    #===============================================================================
    
    if sys.platform == 'win32':
        viewername = "nxv.exe"
        viewerpath = "C:/Program Files (x86)/NxV"
        #viewerpath = "C:\\Program Files (x86)\NxV\\"
        tmp_math = "c:\\tmp\\"
    else:
        viewername = "nxv"
        viewerpath = "/usr/local/bin"
        tmp_math = "/home/marcotegui/tmp/"
        
        vieweroptions = "-RM"
        ext = ".png"
        
	#===============================================================================
	# 			TEST EXISTANCE DU VIEWER
	#===============================================================================
        
        viewer = viewerpath+"/"+viewername
        
	# repertoire temp par defaut (dans le directory temp du user)
        if not os.path.isfile(viewer):
            print("Nxv not present or Nxv path is not correct : ", viewer)
            sys.exit()		
            
	    #===============================================================================
	    #                      RECUPERATON DES ARGUMENTS FONCTION
	    #===============================================================================
            ImageList=[]
            NameList=[]
            
            for item in args:
                if str(type(item))=="<type 'str'>":
                    NameList.append(item+ext)
                else:
                    ImageList.append(item)	
                    
                    while len(ImageList)>len(NameList):
                        
                        NameList.append(os.path.join(tmp_path,"random_%d.png"%random.randint(0,10000)))#tempfile.mkstemp(ext))		
                        
                        for i in range(len(ImageList)):
                            if os.path.isfile(NameList[i]):
                                newdisplay=False
                            else:
                                newdisplay=True
                                break
                            
                            
                            for i in range(len(ImageList)):
                                toto = ImageList[i]
                                if (toto.getDataTypeMax!= 255):
                                    
                                    im8 = Image(toto,"UINT8")
                                    copy(toto,im8)
                                    write(im8,NameList[i])
                                else:
                                    write(ImageList[i],NameList[i])
                                    
                                    
                                if newdisplay is True:
                                    os.spawnv(os.P_NOWAIT, viewer,[viewername,vieweroptions]+NameList)
                                    
                                    v = ImDisplay
                                        
                                        
def ImDisplay3d(im3d, start,M,N,value):
    xsize = im3d.getWidth()
    ysize = im3d.getHeight()
    xsize2 = im3d.getWidth()+2
    ysize2 = im3d.getHeight()+2

    imd = Image(xsize2*M,ysize2*N)
    imd << value
    time = start
    for i in range(N):
        for j in range (M):
            copy(im3d,0,0,time,xsize,ysize,1,imd,j*xsize2,i*ysize2,0)
            time = time + 1

    imd.show()

def highLeveling(immark,imref,imout,nl): 
    imtmp = Image()
    erode(immark, imtmp,nl)
    sup(imtmp,imref,imtmp)
    dualBuild(imtmp,imref,imout,nl)

def lowLeveling(immark,imref,imout,nl):
    imtmp = Image()
    dilate(immark,imtmp,nl)
    inf(imtmp,imref,imtmp)
    build(imtmp,imref,imout,nl)

def leveling(immark,imref,imout,nl):
    imtmp = Image()
    highLeveling(immark,imref,imtmp,nl)
    lowLeveling(immark,imtmp,imout,nl)

# ------------------------------
# Alternate sequential leveling. 
#Filters more than levelings (with AF as marker).
# ------------------------------
def ASF_Leveling(imIn, size, imOut,nl=Morpho.getDefaultSE()):
    #Init images
    imEro = Image(imIn)
    imDil = Image(imIn)
    imTmp1 = Image(imIn)
    imTmp2 = Image(imIn)
    copy(imIn,imEro)
    copy(imIn,imDil)
    print(imIn,imOut)
    copy(imIn,imOut)

    #Alternate sequential leveling
    for i in range(size):
        erode(imEro,imTmp1,nl)
        copy(imTmp1,imEro)

        lowLeveling( imTmp1,imOut, imTmp2,nl)

        dilate(imDil,imTmp1,nl)
        copy(imTmp1,imDil)
        highLeveling(imTmp1, imTmp2, imOut,nl)


def buildAF(imIn, size,imOut,nl=Morpho.getDefaultSE()):
    imTmp = Image(imIn)
    buildOpen(imIn,imTmp,nl(size))
    buildClose(imTmp,imOut,nl(size))

def buildASF(imIn, size, imOut,nl=Morpho.getDefaultSE()):
    imTmp = Image(imIn)
    copy(imIn,imOut)
    for i in range(size):
        buildOpen(imOut,imTmp,nl(i+1))
        buildClose(imTmp,imOut,nl(i+1))

def AF(imIn, size, imOut,nl=Morpho.getDefaultSE()):
    imTmp = Image(imIn)
    open(imIn,imTmp,nl(size))
    close(imTmp,imOut,nl(size))

        
def ASF(imIn, size,imOut,nl=Morpho.getDefaultSE()):
    imTmp = Image(imIn)
    copy(imIn,imOut)
    for i in range(size):
        open(imOut,imTmp,nl(i+1))
        close(imTmp,imOut,nl(i+1))

def overlay(im,imout,color = 0):
    imout.show()
    if (color == 0):
        imout.getViewer().drawOverlay(im)
    else:
        imout.getViewer().drawOverlay(im&color)

def watershedEV(imgra,EVType,nl=Morpho.getDefaultSE()):
    imFineSeg = Image(imgra,"UINT16")
    g = watershedExtinctionGraph(imgra,imFineSeg,EVType)
    return imFineSeg,g

def getEVLevel(imFineSeg,g,    Nregions, imSeg):
    g2 = g.clone() # edges are removed, clone the graph so you can get other partitions in further function calls
    g2.removeLowEdges(Nregions)# removeLowEdges( EdgeWeightType EdgeThreshold)
    graphToMosaic(imFineSeg, g2, imSeg)

def watershedEVI(imgra,EVType,nl=Morpho.getDefaultSE()):
    imEV = Image(imgra,"UINT16")
    watershedExtinction(imgra,imEV,EVType)
    return imEV

def getEVLevelI(imgra,imEV, Nregions, imSeg,nl=Morpho.getDefaultSE()):
    imMark = Image(imEV)
    compare(imEV,">",Nregions,0,imEV,imMark)
    basins(imgra,imMark,imSeg,nl)

def extractChannels(colorim):
    im1,im2,im3 = Image(),Image(),Image()
    copyChannel(colorim,0,im1)
    copyChannel(colorim,1,im2)
    copyChannel(colorim,2,im3)
    return im1, im2, im3


def combineChannels(im1, im2, im3):
    colorout = Image(im1,"RGB")
    copyToChannel(im1,0,colorout)
    copyToChannel(im2,1,colorout)
    copyToChannel(im3,2,colorout)
    return colorout


def disp3D(im3d, start,M,N,value):
    xsize = im3d.getWidth()
    ysize = im3d.getHeight()
    xsize2 = im3d.getWidth()+2
    ysize2 = im3d.getHeight()+2

    imd = Image(xsize2*M,ysize2*N)
    imd << value
    time = start
    for i in range(N):
        for j in range (M):
            copy(im3d,0,0,time,xsize,ysize,1,imd,j*xsize2,i*ysize2,0)
            time = time + 1

    disp([imd])
    return imd


def ImWaterfalls(imgra,imws0,nl,imws1,imtmp):
    compare(imws0,">",0,imgra,255,imtmp)
    dualBuild(imtmp,imgra,imws1,nl)
    copy(imws1,imgra)
    watershed(imgra,imws1,nl)



def ImRandomColor(imIn):
    """ImRandomColor(imIn): returns a color image, with pseudo-random
    values associated to each input value."""
    
    SEED = 448
    
    
    im1 = Image(imIn,"UINT8")
    im2,im3 = Image(im1),Image(im1)
    
    
    lut = GetMap(imIn,8)
    
    
    myMax = maxVal(imIn)#imIn.getDataTypeMax()
    random.seed(SEED)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im1)
    
    random.seed(SEED+1)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im2)
    
    random.seed(SEED+2)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im3)
    lut[0] = 0
    imColor= combineChannels(im1, im2, im3)

    return imColor


def ImRandomColor_fixed_seed(imIn, SEED):
    """ImRandomColor(imIn): returns a color image, with pseudo-random
    values associated to each input value."""
    
    #SEED = random.randint(0,500)#448
    
    
    im1 = Image(imIn,"UINT8")
    im2,im3 = Image(im1),Image(im1)
    
    
    lut = GetMap(imIn,8)
    

    myMax = maxVal(imIn)#imIn.getDataTypeMax()
    random.seed(SEED)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im1)

    random.seed(SEED+1)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im2)

    random.seed(SEED+2)
    for i in range(myMax):
        lut[i]=random.randint(0,255)
    lut[0] = 0
    applyLookup(imIn,lut,im3)
    lut[0] = 0
    imColor= combineChannels(im1, im2, im3)
    
    return imColor


## returns a map with the following depths imDepth_lutDepth
def GetMap(imval,lutDepth):
     ValMax=imval.getDataTypeMax()

     # Allocate map with adapted depths
     if(lutDepth == 8):
         if(ValMax == 255):
             myMap =  Map_UINT8_UINT8()
         elif(ValMax == 65535):
             myMap =  Map_UINT16_UINT8()
         else:
             print ("ERROR(GetMap): BAD TYPES COMBINATION")
     elif(lutDepth == 16):
         if( ValMax == 255):
             myMap =  Map_UINT8_UINT16()
         elif(ValMax == 65535):
             myMap =  Map_UINT16_UINT16()
         else:
             print ("ERROR(GetMap): BAD TYPES COMBINATION")
     elif(lutDepth == 32):
         if(ValMax == 255):
             myMap =  Map_UINT8_UINT32()
         elif(ValMax == 65535):
             myMap =  Map_UINT16_UINT32()
         else:
             print ("ERROR(GetMap): BAD TYPES COMBINATION")
     else:
         print ("ERROR(GetMap): BAD TYPES COMBINATION")
         pdb.set_trace()
     return myMap





def labelWithMeasure(im,imval,imOut,measure_str,nl=Morpho.getDefaultSE()):
     # ----------------------------------------
     # Compute Blobs
     # ----------------------------------------
     imlabel = Image(im,"UINT16")
     label(im,imlabel,nl)
     blobs = computeBlobs(imlabel)

     if(measure_str=="mean"):
         measList=measMeanVals(imval,blobs)
     elif(measure_str=="max"):
         measList=measMaxVals(imval,blobs)
     elif(measure_str=="min"):
         measList=measMinVals(imval,blobs)
     elif(measure_str=="mode"):
         measList=measModeVals(imval,blobs)
     elif(measure_str=="median"):
         measList=measMedianVals(imval,blobs)

     myLUT =  Map_UINT16_UINT16()
     if(measure_str=="mean"):
         for lbl in blobs.keys():
             myLUT[lbl] = int(measList[lbl][0])
     else:#min,max...
         for lbl in blobs.keys():
             myLUT[lbl] = int(measList[lbl])
     imtmp16 = Image(imlabel)

     applyLookup(imlabel,myLUT,imtmp16)
     copy(imtmp16,imOut)

def gradientLAB(colorim,nl=Morpho.getDefaultSE()):
    imgra = Image(colorim.getWidth(),colorim.getHeight())
    gradient_LAB(colorim,imgra,nl)
    return imgra

def gradientHLS(colorim,nl=Morpho.getDefaultSE()):
    imgra = Image(colorim.getWidth(),colorim.getHeight())
    gradient_HLS(colorim,imgra,nl)
    return imgra

def clearOverlay(im):
    im.getViewer().clearOverlay()

def imNorm(im):
    imout = Image(im,"UINT8")
    mymin,mymax = rangeVal(im)
    if(mymin == mymax):
        copy(im,imout)
    else:
        stretchHist(im,mymin,mymax,imout,0,255)
    return imout
