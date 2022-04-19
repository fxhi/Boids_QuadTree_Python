import numpy as np
import random
import math


class Boid():
    def __init__(self, x=0, y=0, velocity=1):
        
        self.position = np.array([x,y])
        
        angle = random.uniform(0, 2*math.pi)
        self.velocity = np.array([math.cos(angle)*velocity, math.sin(angle)*velocity])
        
        factor = 2
        
        self.dist_coh = 30/factor
        self.dist_sep = 20/factor
        self.dist_alig = 30/factor
        
        self.min_velocity = 0.2
        self.max_velocity = 1
        # self.max_velocity = 2.0*10
        self.max_acceleration = 0.1
        
        self.dist_to_boids = np.array([])
    
    def compute_dist_to_boids(self, boids):
        self.dist_to_boid = np.array([np.linalg.norm(boid.position - self.position)
                                               for boid in boids])
        
    def cohesion(self, boids):
        barycentre = np.array([0.,0.])
        count = 0
        for i,boid in enumerate(boids):
            dist = self.dist_to_boid[i]
            if (0 < dist < self.dist_coh):
                barycentre = barycentre + boid.position
                count += 1
                
        if count > 0:
            direction = barycentre/count - self.position
            direction_lenght = np.linalg.norm(direction)
            if  direction_lenght != 0:
                direction = direction/direction_lenght #normalize
                velocity = direction*self.max_velocity
                acceleration = velocity - self.velocity
                
                acceleration_lenght = np.linalg.norm(acceleration)
                if acceleration_lenght > self.max_acceleration:
                    acceleration[0] = acceleration[0]*self.max_acceleration/acceleration_lenght
                    acceleration[1] = acceleration[1]*self.max_acceleration/acceleration_lenght     
                
                return acceleration
        else:
            direction = np.array([0.,0.])
            unit_vector = direction
            
        return unit_vector  # * 1/time_step  = 1, to get a velocity
        # return direction / self.acceleration_limiter_cohesion # * 1/time_step  = 1, to get a velocity
        
    
    def separation(self, boids):
        new_position = np.array([0.,0.])
        for i,boid in enumerate(boids):
            dist = self.dist_to_boid[i]
            if (0 < dist < self.dist_sep):
                new_position = new_position + ( self.position - boid.position)/dist**2
                
        lenght = np.sqrt(new_position[0]**2 + new_position[1]**2)
        if lenght != 0:
            unit_vector = new_position/lenght
            unit_vector *= self.max_velocity
            unit_vector = unit_vector - self.velocity
            
            lenght_unit_vector = np.sqrt(unit_vector[0]**2 + unit_vector[1]**2)
            if lenght_unit_vector > self.max_acceleration:
                unit_vector[0] = unit_vector[0]*self.max_acceleration/lenght_unit_vector
                unit_vector[1] = unit_vector[1]*self.max_acceleration/lenght_unit_vector
                
            return unit_vector
            
            
        return np.array([0.,0.]) # * 1/time_step (=1) to get a velocity
    
    
    def alignment(self, boids):
        new_velocity = np.array([0.,0.], dtype='float64')
        count = 0
        for i,boid in enumerate(boids):
            dist = self.dist_to_boid[i]
            if (0 < dist < self.dist_alig):
                new_velocity = new_velocity + boid.velocity
                count += 1
                
        
        if count > 0 :
            new_velocity = new_velocity/count 
            lenght = np.sqrt(new_velocity[0]**2 + new_velocity[1]**2)
            if lenght != 0:
                unit_vector = new_velocity/lenght
                unit_vector *= self.max_velocity
                unit_vector = unit_vector - self.velocity
                
                lenght_unit_vector = np.sqrt(unit_vector[0]**2 + unit_vector[1]**2)
                if lenght_unit_vector > self.max_acceleration:
                    unit_vector[0] = unit_vector[0]*self.max_acceleration/lenght_unit_vector
                    unit_vector[1] = unit_vector[1]*self.max_acceleration/lenght_unit_vector
                    
                return unit_vector

        else:
            return np.array([0.,0.])
        
    
    def run(self, boids):
        self.compute_dist_to_boids(boids)
        rule1 = self.separation(boids)
        rule2 = self.alignment(boids)
        rule3 = self.cohesion(boids)
        
        acc = rule1*1.5 + rule2 + rule3
        
        if np.linalg.norm(acc) > self.max_acceleration :
            acc *= self.max_acceleration/np.linalg.norm(acc) 
        
        self.velocity = self.velocity + acc
        
        if (np.linalg.norm(self.velocity) > self.max_velocity):
            self.velocity *= self.max_velocity/np.linalg.norm(self.velocity)
            
        self.position = self.position + self.velocity # time_step = 1
        
        
    def run_qtree(self, qtree, delta_t):
        points = []
        qtree.query_radius(self.position, max(self.dist_sep, self.dist_alig, self.dist_coh), points)
        boids = [point.payload for point in points]
        
        self.compute_dist_to_boids(boids)
        
        rule1 = self.separation(boids)
        rule2 = self.alignment(boids)
        rule3 = self.cohesion(boids)
        
        acc = rule1*1.5 + rule2 + rule3
        
        if np.linalg.norm(acc) > self.max_acceleration :
            acc *= self.max_acceleration/np.linalg.norm(acc) 
        
        self.velocity = self.velocity + acc
        
        if (np.linalg.norm(self.velocity) > self.max_velocity):
            self.velocity *= self.max_velocity/np.linalg.norm(self.velocity)
            
        self.position = self.position + self.velocity # time_step = 1
        
        
    
class Flock():
    def __init__(self, n=2, width=640, height=360, quadtree_max_points=4):
        self.boids_number = n
        self.width = width
        self.height = height
        self.delta_t = 1.0/120 #120 FPS
        self.boids = [Boid(self.width/2, self.height/2) for i in range(n)]
        
        self.quadtree_max_points = quadtree_max_points

        self.create_quadtree()
        
        
    def set_time_step(self, delta_t):
        self.delta_t = delta_t
        
        
    def create_quadtree(self):
        self.qtree = QuadTree(Rect(self.width/2, self.height/2, self.width, self.height), max_points = self.quadtree_max_points )
        for boid in self.boids:
            self.qtree.insert(Point(*boid.position, boid))
            # self.qtree.insert(Point(boid.position[0], boid.position[1], boid))

        
    def update_boids(self):
        for boid in self.boids:
            boid.run_qtree(self.qtree, self.delta_t)
            # boid.run(self.boids)
            
        self.create_quadtree()
        
            
    def boundary_condition(self):
        # Impose periodic boundary condition
        for boid in self.boids:
            x,y = boid.position
            x = (x+self.width) % self.width
            y = (y+self.height) % self.height
            boid.position = np.array([x,y])
        
            
    def run(self):
        self.update_boids()
        self.boundary_condition()
        
        
if  __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import time
    import os
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    os.chdir(ROOT_DIR)
    from Quadtree import Rect, QuadTree, Point
    
    n=50
    width, height = 640, 360
    flock = Flock(n, width, height, 4)
    
    P = np.zeros((n,2))
    circle = plt.Circle((0,0), 10, ec='r', color = 'gray', alpha=0.4)
    
    def update(*args):
        start_time = time.time() # start time of the loop
        ax.clear()
        ax.set_xlim(0,width)
        ax.set_ylim(0,height)
        ax.set_xticks([])
        ax.set_yticks([])
        flock.run()
        for i,boid in enumerate(flock.boids):
            P[i] = boid.position
        ax.scatter(P[:,0], P[:,1], s=30, facecolor="red", edgecolor="None", alpha=0.5)
        
        circle = plt.Circle(
            flock.boids[0].position, 
            max(flock.boids[0].dist_alig, flock.boids[0].dist_coh, flock.boids[0].dist_sep),
            ec='r', color = 'gray', alpha=0.4)
        
        ax.add_patch(circle)
        flock.qtree.draw(ax)

        delta_t = time.time() - start_time
        flock.set_time_step(delta_t)
        # print("Elasped time : {}".format(delta_t+0.0000001), flush=True) # FPS = 1 / time to process loop
        
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    animation = animation.FuncAnimation(fig, update, interval=10)
    plt.show(block=False)