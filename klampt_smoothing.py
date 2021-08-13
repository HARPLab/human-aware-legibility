import table_path_code as resto

try:
    import OpenGL as ogl
    try:
        import OpenGL.GL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        print('Drat, patching for Big Sur')
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass

import klampt
from klampt import vis
from klampt.model import trajectory
from klampt.model import multipath
from klampt.vis import GLRealtimeProgram
import cv2

# #               red         green       yellow          light yellow    grey        light blue      blue            
# PATH_COLORS = [(138,43,226), (0,201,87), (0,255,255), (0,128,128), (100,100,100), (255,64,64), (255,10,10)]

def draw_path(restaurant, title, traj, prefix):
    img = restaurant.get_img()

    path = traj.milestones

    path_color = (255,64,64) #(0,201,87)
    dot_color =  (255,10,10) # (0,255,255)

    for i in range(len(path) - 1):
        a = tuple(path[i])
        b = tuple(path[i + 1])

        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        
        cv2.line(img, a, b, path_color, thickness=5, lineType=8)
        cv2.circle(img, a, 6, dot_color, 6)

    img = cv2.flip(img, 0)
    cv2.imwrite(prefix + title + '.png', img)
    print("Drawing completed for " + title)


class GLTest(GLRealtimeProgram):
    """Define hooks into the GUI loop to draw and update the simulation"""
    def __init__(self,world,sim):
        GLRealtimeProgram.__init__(self,"GLTest")
        self.world = world
        self.sim = sim

    def display(self):
        self.sim.updateWorld()
        self.world.drawGL()
        pass

    def idle(self):
        rfs = sim.controller(0).sensor("RF_ForceSensor")
        print("Sensor values:",rfs.getMeasurements())
        sim.simulate(self.dt)
        return

if __name__ == "__main__":
    world = klampt.WorldModel()
    # res = world.readFile("../../data/hubo_plane.xml")
    # if not res:
    #     raise RuntimeError("Unable to load world")
    # sim = klampt.Simulator(world)
    # print("STARTING vis.run()")
    # vis.run(GLTest(world,sim))
    # print("END OF vis.run()")


    p1 = [[104, 477], [141, 459], [178, 444], [215, 430], [251, 417], [287, 405], [322, 395], [357, 386], [391, 379], [425, 373], [459, 368], [492, 365], [525, 363], [557, 363], [588, 364], [620, 366], [651, 370], [681, 375], [711, 381], [740, 389], [769, 398], [798, 409], [826, 421], [854, 434], [881, 449], [908, 465], [934, 483], [960, 502], [985, 522], [1010, 543], [1035, 567]]
    p2 = [[104, 477], [147, 447], [190, 419], [231, 394], [272, 371], [312, 350], [351, 331], [390, 315], [427, 301], [464, 289], [499, 280], [534, 273], [568, 268], [601, 265], [634, 265], [665, 267], [696, 271], [726, 277], [755, 286], [783, 297], [810, 310], [836, 325], [862, 343], [886, 363], [910, 385], [933, 410], [955, 437], [976, 466], [996, 497], [1016, 531], [1035, 567]]
    p3 = [[104, 477], [124, 447], [145, 419], [167, 394], [190, 371], [213, 350], [237, 332], [262, 315], [288, 301], [314, 290], [341, 280], [369, 273], [397, 268], [427, 266], [457, 265], [487, 267], [519, 271], [551, 278], [584, 286], [617, 297], [652, 310], [687, 326], [722, 343], [759, 363], [796, 386], [834, 410], [873, 437], [912, 466], [952, 497], [993, 531], [1035, 567]]
    p4 = [[104, 477], [146, 446], [187, 418], [228, 392], [268, 369], [307, 348], [345, 329], [383, 313], [420, 298], [456, 286], [491, 277], [525, 269], [559, 264], [592, 262], [624, 261], [656, 263], [686, 267], [716, 274], [745, 282], [774, 293], [801, 307], [828, 322], [854, 340], [879, 361], [904, 383], [928, 408], [950, 435], [973, 464], [994, 496], [1015, 530], [1035, 567]]
    p5 = [[104, 477], [98, 509], [95, 540], [95, 569], [97, 596], [101, 620], [108, 643], [118, 663], [130, 682], [145, 698], [162, 712], [182, 725], [204, 735], [229, 743], [256, 749], [286, 753], [318, 755], [353, 755], [390, 753], [430, 749], [472, 742], [517, 734], [565, 724], [615, 711], [667, 697], [722, 680], [779, 662], [839, 641], [902, 618], [967, 593], [1035, 567]]

    # path = multipath.MultiPath(p1)

    generate_type   = resto.TYPE_EXP_SINGLE
    r               = resto.Restaurant(generate_type)

    prefix = 'path_optimization/sqrt-'
    prefix = 'path_optimization/simple-'
    dt = 1

    mid_path = int(len(p1) / 2)
    path = [p1[0], p1[mid_path], p1[-1]][::-1]
    # path = [[104, 477, 90], p1[mid_path], [1035, 567, 0]]

    title = 'linear'
    traj = trajectory.path_to_trajectory(path, speed=1, timing='sqrt-L2', velocities='auto')
    dt = int(len(traj.milestones)) * .001

    print(dt)

    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print(traj.milestones)
    print(len(traj.milestones))

    title = 'limited'
    traj = trajectory.path_to_trajectory(path, speed=1, timing='limited', velocities='auto')
    dt = int(len(traj.milestones)) * .001

    print(dt)

    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print(traj.milestones)
    print(len(traj.milestones))

    print("min-jerk")
    title = 'min-jerk'
    traj = trajectory.path_to_trajectory(path, speed=1, velocities='minimum-jerk', timing='sqrt-L2')
    dt = int(len(traj.milestones)) * .001
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print(traj.milestones)
    print(len(traj.milestones))

    exit()

    title = 't1'
    traj = trajectory.path_to_trajectory(path, speed=1, timing='limited')
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print(traj.milestones)
    print(len(traj.milestones))

    print("parabolic")
    title = 't2'
    traj = trajectory.path_to_trajectory(path, speed=1, velocities='parabolic', timing='sqrt-L2')
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print(traj.milestones)
    print(len(traj.milestones))

    title = 'triangular'
    traj = trajectory.path_to_trajectory(path, speed=1, velocities='triangular', timing='sqrt-L2')
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print("auto")
    print(traj.milestones)
    print(len(traj.milestones))
    
    title = 'trapezoid'
    traj = trajectory.path_to_trajectory(path, speed=1, velocities='trapezoidal', timing='sqrt-L2')
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print("trapezoidal")
    print(traj.milestones)
    print(len(traj.milestones))

    title = 'cosine'
    traj = trajectory.path_to_trajectory(path, speed=1, velocities='cosine', timing='sqrt-L2')
    traj = traj.discretize(dt)
    traj.save(prefix + title + '.txt')
    draw_path(r, title, traj, prefix)

    print("cosine")
    print(traj.milestones)
    print(len(traj.milestones))

    # velocities=’auto’, trapezoidal’, ‘triangular’, ‘parabolic’, ‘cosine’, or ‘minimum-jerk’;




    print("got trajectories")


