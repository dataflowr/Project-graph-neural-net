#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>


/// Function from https://stackoverflow.com/a/2736841
char *remove_ext (char* myStr) {
    char *retStr, *lastExt, *lastPath;
    char extSep = '.';
    char pathSep = '/';

    if (myStr == NULL) return NULL;
    if ((retStr = malloc (strlen (myStr) + 1)) == NULL) return NULL;

    strcpy (retStr, myStr);
    lastExt = strrchr (retStr, extSep);
    lastPath = (pathSep == 0) ? NULL : strrchr (retStr, pathSep);

    if (lastExt != NULL) {
        if (lastPath != NULL){
            if (lastPath < lastExt)
                *lastExt = '\0';
        }
        else
            *lastExt = '\0';
    }

    return retStr;
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char folderConfigFilesToRun[] = "./configs/configs_to_run/";
    const char folderConfigFilesComputed[] = "./configs/configs_computed/";
    
    // First we count how many config files we have to run
    int nbConfigFiles = 0;
    DIR *d;
    struct dirent *dir;
    d = opendir(folderConfigFilesToRun);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            char* ext = strrchr(dir->d_name, '.');
            if (ext != NULL && strcmp(ext, ".yaml") == 0)
                nbConfigFiles++;
        }
        closedir(d);
    }

    // Then we creates an array containing the name of the config files to be run
    char **fileList = malloc(nbConfigFiles * sizeof(char *));
    int i = 0;
    d = opendir(folderConfigFilesToRun);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            char* ext = strrchr(dir->d_name, '.');
            if (ext != NULL && strcmp(ext, ".yaml") == 0){
                fileList[i] = strdup(dir->d_name);
                i++;
            }
        }
        closedir(d);
    }

    // Each of the MPI process run their associated config files
    for (i = rank; i < nbConfigFiles; i += size){
        // Creates the log file associated to the current config file
        char logFile[500];
        strcpy(logFile, folderConfigFilesComputed);
        strcat(logFile, remove_ext(fileList[i]));
        strcat(logFile, ".log");

        // Redirects everything printed into the log file
        freopen(logFile,"w",stdout);
        dup2(fileno(stdout), fileno(stderr));

        // Starts the training with the current config file
        char cmd[500] = "python3 commander.py train with ";
        strcat(cmd, folderConfigFilesToRun);
        strcat(cmd, fileList[i]);
        system(cmd);

        // Closes the redirection to the log file
        fclose (stdout);
        
        // Relocates the computed file in the computed config files folder
        char fileSrc[500];
        strcpy(fileSrc, folderConfigFilesToRun);
        strcat(fileSrc, fileList[i]);
        char fileDest[500];
        strcpy(fileDest, folderConfigFilesComputed);
        strcat(fileDest, fileList[i]);
        rename(fileSrc, fileDest);
    }


    MPI_Finalize();
    return 0;
}
