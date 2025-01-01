# Churnobyl

[![Deploy](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml)
<!--[![Tests](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml)-->


Super excited to say the project is now complete and has achieved OPERATION VACATION!! The whole pipeline runs via Github Actions and deploys a docker container on a gcloud instance.

You can use this as a template to run and deploy an MLOps project



## Future additions:

- Minimize human intervention. The pipeline must run automatically, if the production metric goes below a certain threshold
- Adding storing of prediction endpoint, right now it does not store any data
- Adding a flag endpoint to capture wrong predictions submitted by users
- Using self-hosted runners instead of Github appointed, to gain more control over functioning and logging of the pipeline
- Model training doesn't support `StackingClassifier`
- Tuning `VotingClassifier` is not available yet

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


### [Contributions](./CONTRIBUTING.md)

Any help is always welcomed. The project is open-sourced. The key features that are needed to be updated are marked as TODO in readme as well as in code. Some issues according to the authors of the project are highlighted in the README itself. If you think there can be any other improvement, please make a PR or an issue, and I'll go over it as soon as possible.


### Steps to develop:

- Join the org!
- Create a PR
- Get the keys
- Notify owners and reviewers of repo/project
- Push code to branch, some tests must be passes by the branch
[LICENCE](LICENCE)

Refs:

- Pulumi seems like a great automation tool to "up" the infra, would have to look into ECS first, but lets see. Very hopeful about this
- https://www.youtube.com/watch?app=desktop&v=ZaTVXLuCXQ8
- https://stackoverflow.com/questions/19555525/saving-plots-axessubplot-generated-from-python-pandas-with-matplotlibs-savefi
- https://www.youtube.com/watch?v=iqrS7Q174Ac & https://www.youtube.com/watch?v=h5wLuVDr0oc&t=397s
- https://github.com/vb100/github-action-to-ecr/blob/main/.github/workflows/main.yml & https://www.youtube.com/watch?v=m1OFz_Y9bYo & https://www.youtube.com/watch?v=2qE4Kd1Lxmc
- https://github.com/ricardo-vaca/serverless-fastapi
- https://www.ravirajag.dev/blog/mlops-monitoring
- https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
- https://www.eliasbrange.dev/posts/deploy-fastapi-on-aws-part-1-lambda-api-gateway/
- https://github.com/eliasbrange/aws-fastapi/tree/main/lambda-api-gateway
- https://blog.searce.com/fastapi-container-app-deployment-using-aws-lambda-and-api-gateway-6721904531d0
- https://www.youtube.com/watch?v=rpVLOVeky6A&ab_channel=VincentStevenson
- https://www.youtube.com/watch?v=UXMTOBkdvMY
- https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
- https://github.blog/2022-05-09-supercharging-github-actions-with-job-summaries/
- https://cml.dev/doc/usage
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push.html
- https://medium.com/akava/deploying-containerized-aws-lambda-functions-with-terraform-7147b9815599
- https://developer.hashicorp.com/terraform/tutorials/aws/lambda-api-gateway
- https://www.youtube.com/watch?v=VYk3lwZbHBU&ab_channel=TalhaAnwar
- https://www.baeldung.com/ops/dockerfile-env-variable
- https://www.youtube.com/watch?v=9ciujhGdAOs&ab_channel=TechplanetAI
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.html
- https://www.youtube.com/watch?v=M91vXdjve7A&ab_channel=BeABetterDev
- https://octopus.com/blog/githubactions-docker-ecr
- https://aws.amazon.com/blogs/compute/using-aws-lambda-extensions-to-send-logs-to-custom-destinations/
- https://madewithml.com/courses/mlops/testing/
- https://neptune.ai/blog/ml-model-testing-teams-share-how-they-test-models
- https://github.com/whylabs/whylogs/tree/mainline
- https://whylogs.readthedocs.io/en/stable/examples/integrations/flask_streaming/flask_with_whylogs.html
- https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/
- https://betterprogramming.pub/deploying-ml-models-using-aws-lambda-and-api-gateway-f1f349515c81
- https://aws.amazon.com/blogs/machine-learning/deploy-a-machine-learning-inference-data-capture-solution-on-aws-lambda/
- https://blog.devgenius.io/how-to-put-an-ml-model-into-production-dc3a99eeb2cc
- https://aws.amazon.com/blogs/machine-learning/build-a-ci-cd-pipeline-for-deploying-custom-machine-learning-models-using-aws-services/
- https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://towardsai.net/p/l/model-monitoring-dashboards-made-easy-1-3%3Famp%3D1&ved=2ahUKEwjF06zUrI2AAxVQcGwGHekJDvEQFnoECCcQAQ&usg=AOvVaw1a1kNPwMKpByjAb8PkguQ9
- https://docs.arize.com/arize/quickstart
- https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://neptune.ai/blog/ml-model-monitoring-best-tools&ved=2ahUKEwjF06zUrI2AAxVQcGwGHekJDvEQFnoECCUQAQ&usg=AOvVaw0cWnCVHcV7-lAU9ivcr-lf
- https://retool.com/templates/machine-learning-model-monitoring-dashboard
- https://docs.dominodatalab.com/en/5.3/user_guide/715969/model-monitoring/
- https://stackoverflow.com/questions/65602056/how-to-set-and-access-environment-variables-in-python-file-for-streamlit-app
- https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact

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
