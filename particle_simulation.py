import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
class Particle:
    
    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius
        self.styles = styles
        if not self.styles:
            self.styles = {'edgecolor': 'b', 'fill': False}
    @property
    def x(self):
        return self.r[0]
    
    @x.setter
    def x(self, value):
        self.r[0] = value
        
    @property
    def y(self):
        return self.r[1]
    
    @y.setter
    def y(self, value):
        self.r[1] = value
        
    @property
    def vx(self):
        return self.v[0]
    
    @vx.setter
    def vx(self, value):
        self.v[0] = value
        
    @property
    def vy(self):
        return self.v[1]
    
    @vy.setter
    def vy(self, value):
        self.v[1] = value
        
    def overlaps(self, other):
        return np.hypot(*(self.r - other.r)) < self.radius + other.radius
    
    def draw(self, ax):
        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle
    
    def advance(self, dt):
        self.r += self.v * dt
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
            
        if self.x + self.radius > 1:
            self.x = 1-self.radius
            self.vx = -self.vx
            
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
            
        if self.y + self.radius > 1:
            self.y = 1-self.radius
            self.vy = -self.vy
            
            
class Simulation:
    def __init__(self, n, radius=0.01, styles=None):
        self.init_particles(n, radius, styles)
        
    def init_particles(self, n, radius, styles=None):
        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)
        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):
            while True:
                x, y = rad + (1 - 2*rad) * np.random.random(2)
                vr = 0.1 * np.random.random() + 0.05
                vphi = 2*np.pi * np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                particle = Particle(x, y, vx, vy, rad, styles)
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break
                
    def handle_collisions(self):
        def change_velocities(p1, p2):
            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2
            
        # We're going to need a sequence of all of the pairs of particles when
        # we are detecting collisions. combinations generates pairs of indexes
        # into the self.particles list of Particles on the fly.
        pairs = combinations(range(self.n), 2)
        for i,j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                change_velocities(self.particles[i], self.particles[j])
                
    def advance_animation(self, dt):
        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
        self.handle_collisions()
        return self.circles
    
    def advance(self, dt):
        for i, p in enumerate(self.particles):
            p.advance(dt)
        self.handle_collisions()
        
    def init(self):
        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax))
        return self.circles
    
    def animate(self, i):
        self.advance_animation(0.01)
        #plt.savefig(str(i) + ".png")
        return self.circles
    
    def do_animation(self, save=False):
        fig, self.ax = plt.subplots()
        for s in ['top','bottom','left','right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                               frames=800, interval=2, blit=True)
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=100, bitrate=1800)
            anim.save('collision.mp4', writer=writer)
        else:
            plt.show()
            
if __name__ == '__main__':
    nparticles = 10
    radii = (np.random.random(nparticles)*0.03+0.02) * 0.1
    styles = {'edgecolor': 'C0', 'linewidth': 1, 'fill': None}
    sim = Simulation(nparticles, radii, styles)
    sim.do_animation(save=True)