from scipy.optimize import minimize
from argparse import ArgumentParser
from PyMoCapViewer import MoCapViewer

import pandas as pd
import numpy as np
import math
import yaml
import time

class PoseCorrector:

    def __init__(self, estimatedKeypointsFile, referenceMarkersFile="", numberKeypoints=0, skeletonModelFile="", errorThreshold=0.1, output="output.csv", show=False):
        
        # Initialize the class with the given inputs
        self.est_KP= pd.read_csv(estimatedKeypointsFile, delimiter=";") # import estimated keypoints obtained from OpenPose
        self.ref_KP = pd.read_csv(referenceMarkersFile, delimiter=";") # import reference keypoints (markers)
        if show:
            self.render = MoCapViewer(sampling_frequency=60)
            self.render.add_skeleton(self.est_KP, skeleton_connection="gopro2", color="red")
        self.model = self.load_model(skeletonModelFile) # import skeleton model
        self.num_KP = numberKeypoints
        self.tolerance = errorThreshold
        self.nFrame = len(self.est_KP)
        self.outputFilename = output
        self.scope = 100
    
    def show(self):
        self.render.show_window()
    
    def start(self):
        # Repeat for each frame
        for i in range(self.nFrame):
            print(f"{math.trunc((i*100)/self.nFrame)}%", end="\r")
            AllSegmentsAreCorrect = False
            #Repeat until all segment lengths are correct
            while not AllSegmentsAreCorrect:
                AllSegmentsAreCorrect = True
                # For each segment
                for _, segment in self.model.items():
                    # Check if the measurement of the segment's length is correct
                    if not self.isSegmentLengthCorrect(segment["couple"], segment["length"], i):
                        AllSegmentsAreCorrect = False
                        j1 = segment["couple"][0]
                        j2 = segment["couple"][1]
                        l = segment["length"]
                        #print(f"segment = {segment}")
                        if self.iskeypointPoseCorrect(j2,j1,i):
                            #print("correcting joint 1")
                            self.correctKeypointPose(j1,j2,l, i, False)
                            if not self.inTheScope(j2,i):
                                self.correctKeypointPose(j2,j1,l, i, False)
                                    
                        elif self.iskeypointPoseCorrect(j1,j2,i):
                            #print("correcting joint 2")
                            self.correctKeypointPose(j2,j1,l, i, False)
                            if not self.inTheScope(j1,i):
                                self.correctKeypointPose(j1,j2,l, i, False)
                                
                        else :
                            if self.inTheScope(j1,i):
                                if self.inTheScope(j2,i) and self.isProximal(j2):
                                    self.correctKeypointPose(j1,j2,l, i, False)
                                else:
                                    self.correctKeypointPose(j2,j1,l, i, False)
                                #print("proximal")
                            elif self.inTheScope(j2,i):
                                self.correctKeypointPose(j1,j2,l, i, False)
                                #print("proximal")
                                
                            else:    
                                self.correctKeypointPose(j1,j2,l, i, True)
                                self.correctKeypointPose(j2,j1,l, i, True)
                                #print("correctall")
                            i=max(i-1,0)
                            break


        self.saveDataFrame(self.est_KP, self.outputFilename)
    
    

        
    
    def correctKeypointPose(self, jointToCorrect, jointAsReference, targetLength, index, MarkerRefTriggered):
        
        pi = jointToCorrect*3
        pf = pi+3
        
        initial_guess = self.ref_KP.iloc[index, pi:pf]
        bounds = (((initial_guess[0] - self.scope), (initial_guess[0] + self.scope)),\
        ((initial_guess[1] - self.scope), (initial_guess[1] + self.scope)),\
        ((initial_guess[2] - self.scope), (initial_guess[2] + self.scope)))
        
        if MarkerRefTriggered:
            constants = self.ref_KP.iloc[index, jointAsReference*3:jointAsReference*3+3]
            self.est_KP.iloc[index, pi:pf] = self.optimization(constants, targetLength, initial_guess, bounds)  
            
        else:
            constants = self.est_KP.iloc[index, jointAsReference*3:jointAsReference*3+3]
            
            referenceVector = np.array(self.ref_KP.iloc[index, jointAsReference*3:jointAsReference*3+3]) - np.array(initial_guess)
            initial_guess = self.optimization(constants, targetLength, initial_guess, bounds)

            self.scope = 30
            bounds = (((initial_guess[0] - self.scope), (initial_guess[0] + self.scope)),\
            ((initial_guess[1] - self.scope), (initial_guess[1] + self.scope)),\
            ((initial_guess[2] - self.scope), (initial_guess[2] + self.scope)))
            self.scope = 100
            
            self.est_KP.iloc[index, pi:pf] = self.optimizationparallel(constants, referenceVector ,initial_guess, bounds )
            self.ref_KP.iloc[index, pi:pf] = self.est_KP.iloc[index, pi:pf]
        

        
    def optimization(self, constants, targetLength, initial_guess, bounds):
        # Define the equation to optimize
        def objective_function(variables, consts, length):
            x, y, z = variables
            a, b, c = consts
            return (abs(math.sqrt(math.pow(x-a, 2) + math.pow(y-b, 2) + math.pow(z-c, 2)) - length))
            
        result = minimize(objective_function, initial_guess, args=(constants, targetLength,), bounds=bounds)
        return result.x
    
    def optimizationparallel(self, constants, referenceVector ,initial_guess, bounds ):
    
        def objective_function(variables, consts, referenceVector):
            x, y, z = variables
            a, b, c = consts
            reference_line = [referenceVector[i] / np.linalg.norm(referenceVector) for i in range(3)]
            return abs(np.sum(reference_line - np.array([[a-x, b-y, c-z] / np.linalg.norm([a-x, b-y, c-z])])))
        
        result = minimize(objective_function, initial_guess, args=(constants, referenceVector,), bounds=bounds)
        return result.x
    
    def iskeypointPoseCorrect(self, keypoint, excludedKeypoint, index):
        for _, segment in self.model.items():
            couple = segment["couple"]
            if (keypoint in couple) and (excludedKeypoint not in couple):
                if  (self.isSegmentLengthCorrect(couple, segment["length"], index)):
                    return True
        return False
    
    def isSegmentLengthCorrect(self, segmentCouple, segmentLength, index):
        keypoint1 = self.est_KP.iloc[index, segmentCouple[0]*3:segmentCouple[0]*3+3] 
        keypoint2 = self.est_KP.iloc[index, segmentCouple[1]*3:segmentCouple[1]*3+3] 
        distance = self.calculate_distance(keypoint1, keypoint2)
        if abs(distance / segmentLength - 1) <= self.tolerance:
                    return True
        
        return False
    
    def inTheScope(self, p1, index):
        estPt = self.est_KP.iloc[index, p1*3:p1*3+3] 
        refPt = self.ref_KP.iloc[index, p1*3:p1*3+3] 
        distance = self.calculate_distance(estPt, refPt)
        if distance <= self.scope:
            return True
        else: return False
    
    def isProximal(self, j):
        k=0
        for _, segment in self.model.items():
            if j in segment["couple"]:
                k+=1
        if k>1:
            return True
        return False
        
    def calculate_distance(self, p1, p2):
        return math.sqrt(math.pow(p2[0]-p1[0], 2) + math.pow(p2[1]-p1[1], 2) + math.pow(p2[2]-p1[2], 2))
    
    def load_model(self, file_path):
        # Load the skeletal strucutre model
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
            
    def saveDataFrame(self, data, file_path):
        data.to_csv(file_path, sep=";", encoding='utf-8')
    

def parse():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", help="L'emplacement de fichier de mouvement open pose.", required=True)
    parser.add_argument("-r", "--reference", help="L'emplacement de fichier des marqueurs.")
    parser.add_argument("-n", "--number_joints", type=int, help="Nombre des points d'intérêt")
    parser.add_argument("-m", "--model", help="L'emplacement de fichier de model de squelette")
    parser.add_argument("-t", "--tolerance", help="Tolerance à l'erreur", type=float)
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--show", action="store_true", help="Only display the kinematic of source file")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    inputs = parse()
    # output
    pc = PoseCorrector(inputs.source,\
                       inputs.reference,\
                       inputs.number_joints,\
                       inputs.model,\
                       inputs.tolerance,\
                       inputs.output, inputs.show)  
    if (inputs.show):
        pc.show()
    else:
        pc.start()
    
