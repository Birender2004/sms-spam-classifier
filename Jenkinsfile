pipeline {
    agent any

    stages {

        stage('Clone Code') {
            steps {
                git branch: 'main', url: 'https://github.com/Birender2004/sms-spam-classifier.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t spam-app .'
            }
        }

        stage('Start Minikube') {
            steps {
                bat 'minikube start'
            }
        }

        stage('Load Image into Minikube') {
            steps {
                bat 'minikube image load spam-app'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                bat 'kubectl apply -f deployment.yaml'
                bat 'kubectl apply -f service.yaml'
            }
        }

        stage('Restart Deployment') {
            steps {
                bat 'kubectl rollout restart deployment spam-app'
            }
        }

        stage('Verify Deployment') {
            steps {
                bat 'kubectl get pods'
                bat 'kubectl get svc'
            }
        }
    }
}
