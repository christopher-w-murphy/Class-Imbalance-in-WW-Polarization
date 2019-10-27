import csv
import os
import subprocess

import pylhe


def get_targets(import_file, export_file):
    events = pylhe.read_lhe(import_file)

    with open(export_file, 'w') as f:
        writes = csv.writer(f, delimiter=',')
        for event in events:
            eventlist = []
            for particle in event.particles:
                if abs(particle.id) == 24:
                    eventlist += [particle.spin]
            writes.writerow(eventlist)


def get_features(import_file, export_file):
    events = pylhe.read_lhe(import_file)

    with open(export_file, 'w') as f:
        writes = csv.writer(f, delimiter=',')
        for event in events:
            eventlist = []
            for particle in event.particles:
                if particle.status == 1:
                    eventlist += [particle.px, particle.py, particle.pz, particle.e]
            writes.writerow(eventlist)


def orchestrator(input_dir, output_dir):
    dirs = os.listdir(input_dir)
    runs = [dir for dir in dirs if dir.startswith('run')]
    for run in runs:
        import_file = os.path.join(input_dir, run, 'unweighted_events.lhe')
        if not os.path.isfile(import_file):
            subprocess.call(['gunzip', os.path.join(input_dir, run, 'unweighted_events.lhe.gz')])
        export_file = os.path.join(output_dir, run + '.csv')
        if run.endswith('decayed_1'):
            get_features(import_file, export_file)
        else:
            get_targets(import_file, export_file)
        subprocess.call(['gzip', import_file])

if __name__ == '__main__':
    input_dir = '/Users/christopherwmurphy/Documents/MG5_aMC_v2_6_6/runs/same_sign_ww/Events'
    output_dir = '/Users/christopherwmurphy/Documents/Research/NNtest/Class-Imbalance-in-WW-Polarization/raw_data'
    orchestrator(input_dir, output_dir)
