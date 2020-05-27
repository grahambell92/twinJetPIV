# Script to copy case data to massive
# Written by Graham Bell 24/10/2016

import os
import paramiko
import PIVCasesInfo
from PIVCasesInfo import cases
import os.path

# Rsync command for uploading scripts.
# rsync -arv --include='*'{/,.py} --exclude='POD*' --exclude='*' /home/graham/GBellPhD/ExpCampaign0/PythonRoutines/PIVRoutines graham_bell@m2-login2.massive.org.au:/home/graham_bell/NCIg75_scratch/gbell/ --prune-empty-dirs --copy-links

# Equiviant to mkdir -p recursive mkdir
def mkdir_p(sftp, remote_directory):
    """Change to this directory, recursively making new folders if needed.
    Returns True if any folders were created."""
    if remote_directory == '/':
        # absolute path so change directory to root
        sftp.chdir('/')
        return
    if remote_directory == '':
        # top-level relative directory must exist
        return
    try:
        sftp.chdir(remote_directory) # sub-directory exists
    except IOError:
        dirname, basename = os.path.split(remote_directory.rstrip('/'))
        mkdir_p(sftp, dirname) # make parent directories
        sftp.mkdir(basename) # sub-directory missing, so created it
        sftp.chdir(basename)
        return True


ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))

if False:
    server = 'm2-login2.massive.org.au'
    username = 'graham_bell'
    password = '********'
    massiveBaseFolderPath = '/home/graham_bell/NCIg75_scratch/gbell/twinJetExp0/'


if True:
    server = 'm3-dtn.massive.org.au'
    username = 'grahamb'
    password = '*********'
    massiveBaseFolderPath = '/home/grahamb/le11_scratch/gbell/twinJetExp0/'

#ssh.connect(server, username=username, password=password)

#sftp = ssh.open_sftp()
# Want to copy h5 files from each set.
[case1, case2, case3, case4, case5, case6, case7, case8, case9] = cases
# Just do the last set of case1 as velocity
cases = [case1, case4]

for caseIndex, case in enumerate(cases):
    localH5Files = [PIVCasesInfo.setPath(PIVset) + 'SUBD/' + 'compiledNC.h5' for PIVset in case['sets']]
    massiveSetPaths = [massiveBaseFolderPath +
                       PIVCasesInfo.setPath(PIVset, False) + 'SUBD/' for PIVset in case['sets']]

    transport = paramiko.Transport(server)

    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for fileIndex, (localPath, remotePath) in enumerate(zip(localH5Files, massiveSetPaths)):
        # Check if directory exists and create if needed
        # Create the directory
        print('Case', caseIndex, 'of', len(cases), 'File', fileIndex, 'of', len(localH5Files))
        print('Transfering file', localPath)
        mkdir_p(sftp, remotePath)
        remoteFilePath = remotePath + 'compiledNC.h5'
        print('To:', remoteFilePath)
        sftp.put(localPath, remoteFilePath)
        print('Done.')
    # upload codes:

sftp.close()
ssh.close()


