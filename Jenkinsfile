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

        stage('Load Image into Minikube') {
            steps {
                bat 'minikube image load spam-app'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                bat '''
                kubectl --kubeconfig="%USERPROFILE%\\.kube\\config" apply -f deployment.yaml
                kubectl --kubeconfig="%USERPROFILE%\\.kube\\config" apply -f service.yaml
                '''
            }
        }

        stage('Restart Deployment') {
            steps {
                bat '''
                kubectl --kubeconfig="%USERPROFILE%\\.kube\\config" rollout restart deployment spam-app
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                bat '''
                kubectl --kubeconfig="%USERPROFILE%\\.kube\\config" get pods
                kubectl --kubeconfig="%USERPROFILE%\\.kube\\config" get svc
                '''
            }
        }
    }
}
