import numpy as np
import cv2 as cv
import asyncio
import random
import time

MapSize = (20, 10)
DisplayScale = 24
UIHeight = 40

Map = np.zeros(MapSize).astype(np.uint8)
StartX = round(Map.shape[1] / 2 - 1)
CurrentScore = 0

Tile = cv.imread("Tile.png")
bImg = cv.imread("UiBorder.png")

Tetra = None                                #Global reference to active tetramino
TetraShapes = []
TetraShapes.append([
[1],
[1],
[1],
[1]])
TetraShapes.append([
[2,2],
[2,2]])
TetraShapes.append([
[0,3,0],
[3,3,3]])
TetraShapes.append([
[0,4],
[0,4],
[4,4]])
TetraShapes.append([
[5,0],
[5,0],
[5,5]])
TetraShapes.append([
[0,6,6],
[6,6,0]])
TetraShapes.append([
[7,7,0],
[0,7,7]])

ScoreValues = (0, 40, 100, 300, 1200)
ColorMap = [
    (90, 60, 30),
    (180, 180, 180),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

class Tetramino:
    def __init__(s, tiles = None):
        if tiles is None:
            r = random.randrange(0, len(TetraShapes))
            s.tiles = np.asarray(TetraShapes[r], dtype = np.uint8)
        else:
            s.tiles = np.asarray(tiles, dtype = np.uint8)
        s.h = s.tiles.shape[0]
        s.w = s.tiles.shape[1]
        s.y = 0
        s.x = StartX - round((s.w - 1) / 2)
        
    def MoveY(s, yMove = 1):
        if s.y + s.h >= MapSize[0]:         #Hit bottom, stop
            s.Sit()
        else:
            tArea = Map[yMove+s.y:yMove+s.y+s.h, s.x:s.x+s.w]
            if tArea.shape == s.tiles.shape:
                if (tArea * s.tiles).any():
                    if s.y <= 0:
                        Lose()
                    s.Sit()
                else:
                    s.y += yMove
            else:
                s.Sit()
    
    def MoveX(s, xMove):
        if s.x + xMove < 0 or s.x + s.w + xMove > Map.shape[1]: return
        
        tArea = Map[s.y:s.y+s.h, xMove+s.x:xMove+s.x+s.w]
        if tArea.shape != s.tiles.shape: return
        if (tArea * s.tiles).any(): return
        
        s.x += xMove
    
    def Rotate(s, Dir = cv.ROTATE_90_CLOCKWISE):
        assert Dir == cv.ROTATE_90_CLOCKWISE or Dir == cv.ROTATE_90_COUNTERCLOCKWISE
        
        oC = np.asarray([s.y + s.h / 2, s.x + s.w / 2])
        s.w, s.h = s.h, s.w
        nC = np.asarray([s.y + s.h / 2, s.x + s.w / 2])
        dC = (nC - oC).round()
        
        s.tiles = cv.rotate(s.tiles, Dir)
        if s.x + s.w > Map.shape[1]: s.MoveX(Map.shape[1] - (s.x + s.w) )
        
        return False
    
    def Sit(s):                             #Draws self onto background removes global reference
        global Map
        global Tetra
        
        tmp = np.zeros(Map.shape, dtype=np.uint8)
        tmp[s.y:s.y+s.h,s.x:s.x+s.w] = s.tiles
        Map = cv.bitwise_or(Map, tmp)
        Tetra = None
    
    def Draw(s, im):                        #Draws self to specified image
        tmp = np.zeros(im.shape, dtype=np.uint8)
        tmp[s.y:s.y+s.h,s.x:s.x+s.w] = s.tiles
        return cv.bitwise_or(im, tmp)

def Score(rowStart, rows):
    global Map
    global CurrentScore
    
    top = np.zeros((rows, Map.shape[1]), dtype=np.uint8)
    mid = Map[0:rowStart + 1]
    bot = Map[rowStart+rows + 1:Map.shape[0]]
    
    CurrentScore += ScoreValues[rows]
    Map = cv.vconcat([top, mid, bot])
    return None

def Main():
    global Map
    global Tetra
    
    while True:
        #Check if we score full rows
        rowFullness = []
        for i in range(Map.shape[0]):
            rowFullness.append((Map[i] > 0).all())
        fullCount = 0
        for i in range(len(rowFullness) - 1, -1, -1):   #I HATE THIS(Read array backwards)
            if rowFullness[i] and fullCount < len(ScoreValues):
                fullCount += 1
            else:
                if fullCount > 0: Score(i, fullCount)
                fullCount = 0
        
        #Handle input and wait
        key = cv.waitKey(750)
        if Tetra and key > 0:
            char = chr(key).lower()
            match char:
                case 'a':
                    Tetra.MoveX(-1)
                case 'd':
                    Tetra.MoveX( 1)
                case 'w':
                    Tetra.Rotate()
                case 's':
                    Tetra.MoveY()
                case ' ':
                    while Tetra:
                        Tetra.MoveY()
        
        if Tetra: Tetra.MoveY()
        
        #Choose new tetramino
        if Tetra == None:
            Tetra = Tetramino()
        
        #Render Map
        render = Map.copy()
        if Tetra != None: render = Tetra.Draw(render)
        
        renderCol = cv.cvtColor(render, cv.COLOR_GRAY2BGR)
        #ColorMap[0] =  (random.randrange(30, 90), random.randrange(30, 90), random.randrange(30, 90))
        for i in range(0, len(ColorMap)):
            renderCol[render == i] = ColorMap[i]
        renderCol = cv.resize(renderCol, None, fx=DisplayScale, fy=DisplayScale, interpolation=cv.INTER_AREA)
        renderCol = cv.bitwise_and(renderCol, tMap)
        #renderCol = cv.bitwise_and(renderCol, Grid)
        
        #Render UI
        UI = UIBack.copy()
        cv.putText(UI, f"Score: {CurrentScore}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        #Combine and display
        Display = cv.vconcat((UI, renderCol))
        cv.imshow("NotTetris V0.1", Display)

def UnsliceImg(TSize, im, border, bScale = 1):
    out = np.zeros((int(TSize[0] / bScale), int(TSize[1] / bScale), 3), dtype = np.uint8)
    iW = out.shape[1] - border * 2      #Interior width
    iH = out.shape[0] - border * 2      #Interior height    
    #Corners
    out[0:border, 0:border] = im[0:border, 0:border]
    out[0:border, out.shape[1]-border:out.shape[1]] = im[0:border, im.shape[1]-border:im.shape[1]]
    out[out.shape[0]-border:out.shape[0], 0:border] = im[im.shape[0]-border:im.shape[0], 0:border]
    out[out.shape[0]-border:out.shape[0], out.shape[1]-border:out.shape[1]] = im[im.shape[0]-border:im.shape[0], im.shape[1]-border:im.shape[1]]
    #Edges
    out[border:out.shape[0]-border, 0:border] = cv.resize(im[border:im.shape[0]-border, 0:border], (border, iH), interpolation=cv.INTER_NEAREST)
    out[border:out.shape[0]-border, out.shape[1]-border:out.shape[1]] = cv.resize(im[border:im.shape[0]-border, im.shape[1]-border:im.shape[1]], (border, iH), interpolation=cv.INTER_NEAREST)
    out[0:border, border:out.shape[1]-border] = cv.resize(im[0:border, border:im.shape[1]-border], (iW, border), interpolation=cv.INTER_NEAREST)
    out[out.shape[0]-border:out.shape[0], border:out.shape[1]-border] = cv.resize(im[im.shape[0]-border:im.shape[0], border:im.shape[1]-border], (iW, border), interpolation=cv.INTER_NEAREST)
    #Center
    out[border:out.shape[0]-border, border:out.shape[1]-border] = cv.resize(im[border:im.shape[0]-border, border:im.shape[1]-border], (iW, iH), interpolation=cv.INTER_NEAREST)
    return cv.resize(out, (TSize[1], TSize[0]), interpolation=cv.INTER_NEAREST)

#Preprocessing some things
#Prepare UI border
UIBack = UnsliceImg((UIHeight, Map.shape[1] * DisplayScale), bImg, 5)
#Prepare tile background
TileRS = cv.resize(Tile, (DisplayScale, DisplayScale), interpolation=cv.INTER_AREA)
tMap = np.full((Map.shape[0] * DisplayScale, Map.shape[1] * DisplayScale, 3), 255, dtype=np.uint8)
for y in range(0, Map.shape[0] * DisplayScale, DisplayScale):
    for x in range(0, Map.shape[1] * DisplayScale, DisplayScale):
        tMap[y:y+DisplayScale, x:x+DisplayScale] = TileRS
#Prepare an image for the Grid
Grid = cv.resize(Map.copy(), None, fx=DisplayScale, fy=DisplayScale, interpolation=cv.INTER_AREA)
for y in range(0, Grid.shape[0] + DisplayScale, DisplayScale):
    cv.line(Grid, (0, y), (Grid.shape[1], y), (255, 255, 255), 2)
for x in range(0, Grid.shape[1] + DisplayScale, DisplayScale):
    cv.line(Grid, (x, 0), (x, Grid.shape[0]), (255, 255, 255), 2)
Grid = 255-Grid
Grid = cv.cvtColor(Grid, cv.COLOR_GRAY2BGR)

#cv.imshow("D1", UnsliceImg((400, 800), bImg, 5, 3))
#cv.imshow("D2", tMap)
#cv.imshow("D3", Grid)

def Lose():
    print(f"You Lose... at least you got {CurrentScore} points.")
    quit()

#cv.startWindowThread()
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB == False: Main()