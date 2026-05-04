pipeline {
    agent any

    environment {
        IMAGE_NAME = "spam-app"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/Birender2004/sms-spam-classifier.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t spam-app .'
            }
        }

        stage('Load Image to Minikube') {
            steps {
                sh 'minikube image load spam-app'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
                sh 'kubectl apply -f service.yaml'
            }
        }

        stage('Verify Deployment') {
            steps {
                sh 'kubectl get pods'
                sh 'kubectl get svc'
            }
        }
    }
}