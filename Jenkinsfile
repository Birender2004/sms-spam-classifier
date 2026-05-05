pipeline {
    agent any

    environment {
        KUBECONFIG_PATH = "C:\\Users\\Birender Pal Singh\\.kube\\config"
    }

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

        stage('Load Image into Minikube') {
            steps {
                bat 'minikube image load spam-app'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                bat "kubectl --kubeconfig=%KUBECONFIG_PATH% apply -f deployment.yaml"
                bat "kubectl --kubeconfig=%KUBECONFIG_PATH% apply -f service.yaml"
            }
        }

        stage('Restart Deployment') {
            steps {
                bat "kubectl --kubeconfig=%KUBECONFIG_PATH% rollout restart deployment spam-app"
            }
        }

        stage('Verify Deployment') {
            steps {
                bat "kubectl --kubeconfig=%KUBECONFIG_PATH% get pods"
                bat "kubectl --kubeconfig=%KUBECONFIG_PATH% get svc"
            }
        }
    }
}
