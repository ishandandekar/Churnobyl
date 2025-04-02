# Churnobyl

[![Deploy](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml)


Super excited to say the project is now complete and has achieved OPERATION VACATION!! The whole pipeline runs via Github Actions and deploys a docker container on a gcloud instance.

You can use this as a template to run and deploy an MLOps project



## Future additions:

- Minimize human intervention. The pipeline must run automatically, if the production metric goes below a certain threshold

## Milestones

- **18-7-2023**: The pipeline ran completely from start to end. No errors while adding artifacts to server too!
- **7-8-2023**: The model serving API WORKED!!!
- **19-9-2023**: I finally figured out the `encoder_oe` error. **FINALLY**
- **20-9-2023**: A full forward pass from making the dataframe to preprocessing it, to predicting using a model worked... omfg, i am so done
- **21-9-2023**: DUDE DOCKERIZE A TEMP APPLICATION WITH ALL `WANDB` STUFF WORKED LESSSGOOOOOOO!!!. Just have to work the logging the response json to s3 bucket. Other than that everything is finally done. Also, yeah, this image is taking a lot of space.
- **21-1-2024**: The pipeline's architecture has changed a whole lot from the start. I feel I've learned more than I've worked on this. I have a better understanding on how larger scale applications may (or atleast should) work. The only part that is left is pushing artifacts to various remote storage locations. I've integrated prefect as well.
- **28-1-2024**: MLFlow was a success, lets see where we can go from here
- **28-12-2024**: So I tried to deploy an API via GCP and it worked (after I tried for 4 hours), now I'm thinking the artifacts will be downloaded and packaged into the image so that only necessary things go to image
- **1-1-2025**: API has been deployed! OPERATION VACATION has been achieved!!!!
- **17-1-2025**: Added prediction logging via response JSON storing into a GCP bucket


### [Contributions](./CONTRIBUTING.md)

Any help is always welcomed. The project is open-sourced. The key features that are needed to be updated are marked as TODO in readme as well as in code. Some issues according to the authors of the project are highlighted in the README itself. If you think there can be any other improvement, please make a PR or an issue, and I'll go over it as soon as possible. Although, I ignore tests, because I think they are somewhat pointless to the project, but please make sure the code you put in the commits does actually run


[LICENCE](LICENCE)

### Notes for the report

- The report should have these topics
  - Abstract
  - Motive
  - Problem statement
  - Data
  - Approach for modelling experiments
  - Conclusion
  - Appendix
- https://excalidraw.com/ for diagrams
