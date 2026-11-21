"""Execute a notebook using the current interpreter and save outputs after each cell."""
import argparse
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from common import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('notebook');a=p.parse_args()
    path=ROOT/'notebooks'/a.notebook
    nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=None,kernel_name='ai651-xpu',resources={'metadata':{'path':str(ROOT)}})
    def save(**kwargs):
        nbformat.write(nb,path)
    client.on_cell_executed=save
    try:
        client.execute()
    finally:
        nbformat.write(nb,path)
    print('Executed',path,flush=True)

if __name__=='__main__':
    main()
