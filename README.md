# Churnobyl

[![Deploy](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/deploy.yaml)
[![Tests](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml/badge.svg)](https://github.com/ishandandekar/Churnobyl/actions/workflows/tests.yaml)

[LICENCE](LICENCE)

> [!WARNING]
> This is a work in progress. Until specified, please do not directly use the code. There will be addtition as well as improvements over the time. Use the code only to get inspiration and not for actual production usage.

## [Contributions](./CONTRIBUTING.md)

Any help is always welcomed. The project is open-sourced. The key features that are needed to be updated are marked as TODO in readme as well as in code. Some issues according to the authors of the project are highlighted in the README itself. If you think there can be any other improvement, please make a PR or an issue, and I'll go over it as soon as possible.

## Future additions:

- Add more configurations that can be changed by the "DEV". For example, add config for hyperparameter tuning, like specifying the model and its params
- Minimize human intervention. The pipeline must run automatically, if the production metric goes below a certain threshold
- Up the cloud infra using tools like Terraform or Ansible
- Using self-hosted runners instead of Github appointed, to gain more control over functioning and logging of the pipeline

### Steps to develop locally:

- Join the org!
- Create a PR
- Get the keys
- Notify owners and reviewers of repo/project
- Push code to branch, some tests must be passes by the branch

## Milestone (18-7-2023): The pipeline ran completely from start to end. No errors while adding artifacts to server too!

## Milestone (7-8-2023): The model serving API WORKED!!!

## Milestone (19-9-2023): I finally figured out the `encoder_oe` error. **FINALLY**

## Milestone (20-9-2023): A full forward pass from making the dataframe to preprocessing it, to predicting using a model worked... omfg, i am so done

## Milestone (21-9-2023): DUDE DOCKERIZE A TEMP APPLICATION WITH ALL `WANDB` STUFF WORKED LESSSGOOOOOOO!!!. Just have to work the logging the response json to s3 bucket. Other than that everything is finally done. Also, yeah, this image is taking a lot of space.

### Issues:

- Wandb cant seem to log preprocessors, still have to figure this out
  > Did some experimentation, `wandb` is only logging the new preprocessors, only if I delete the previous versions, idk what other fix there is. But this'll have to do right now. I'll have to figure out the `encoder_oe` stuff before I tackle this.
- There is no code for imputing values in pipeline. **This has been set aside temporalily**
- The API serving code can be still be optimized. There is too much code that might seem to complicate things. Better serving solutions still need to be tested.
- The code for monitoring can be a pain. Creating a branch for the Streamlit dashboard is one of the solutions.
- For setting up configuration variables right now, `.yaml` seems the way to go. Some other ways like using a `.env` file can also be a method that can be benefiticial for setting up AWS credentials locally.
- DEV notes are still needed to be added for future MLEs
- Model training doesn't support `StackingClassifier`

### S3 directory structure

```
churnobyl/
├─ api_logs/
│  ├─ predict_logs/
│  ├─ flag_logs/
├─ flagged/
├─ train_logs/
```

### Ideas for monitoring dashboard:

- Prediction rate
- Monthly/daily frequency of requests
- Flag rate
- alert after flags cross certain threshold
- Add probability data

### TODO:

- [ ] Test out how you can upload data from python to AWS S3 bucket. [link for yt video](https://www.youtube.com/watch?v=vXiZO1c5Sk0)
- [ ] Look into [Hopsworks](https://docs.hopsworks.ai/3.4/concepts/mlops/registry/) for model registry and deployment
- [ ] https://www.youtube.com/watch?v=XEZ7Hx2NrO8 & https://stackoverflow.com/questions/62664183/mlflow-find-model-version-with-best-metric-using-python-code
- [x] https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/ DOES NOT WORK FFS
- [ ] Try another way to package model so that one program downloads the best transformer and predictor and another script just with inference/prediction function this then gets packged into a Docker image.
- [x] Add visualization to plot training results
- [ ] Update documentation for code
- [x] Incorporate `**kwargs` for **func** `engine.vizualize_insights` to shortcut optional args everytime
- [ ] Update tests for this new integration
- [x] Write `exceptions.py` maybe for better error catching
- [ ] Write func to validate `.yaml` conf
- [ ] Another idea is to setup a new separate repository for flagged data and monitoring, this could make things easier for api deployment as well as maintenence.
- [ ] Refer the [prototype](./temp/predict_prototype.py) for the API
- [ ] Look into EKS cluster to display monitoring
- [ ] Create Streamlit dashboard for monitoring
- [ ] Integrate Streamlit to display graphs
- [ ] Write a report on the project explaining all the components

### Refs:

- Pulumi seems like a great automation tool to "up" the infra, would have to look into ECS first, but lets see. Very hopeful about this
- https://www.youtube.com/watch?app=desktop&v=ZaTVXLuCXQ8
- https://stackoverflow.com/questions/19555525/saving-plots-axessubplot-generated-from-python-pandas-with-matplotlibs-savefi
- https://github.com/ricardo-vaca/serverless-fastapi
- https://www.ravirajag.dev/blog/mlops-monitoring
- https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
- https://www.eliasbrange.dev/posts/deploy-fastapi-on-aws-part-1-lambda-api-gateway/
- https://github.com/eliasbrange/aws-fastapi/tree/main/lambda-api-gateway
- https://blog.searce.com/fastapi-container-app-deployment-using-aws-lambda-and-api-gateway-6721904531d0
- https://www.youtube.com/watch?v=rpVLOVeky6A&ab_channel=VincentStevenson
- https://www.youtube.com/watch?v=UXMTOBkdvMY

---

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
