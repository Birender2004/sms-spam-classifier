pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Birender2004/sms-spam-classifier.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t spam-app .'
            }
        }

        stage('Deploy Info') {
            steps {
                sh 'echo "Docker image built successfully. Deployment handled locally via Kubernetes."'
            }
        }
    }
}
