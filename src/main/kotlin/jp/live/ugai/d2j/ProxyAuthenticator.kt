package jp.live.ugai.d2j

import java.net.Authenticator
import java.net.PasswordAuthentication

class ProxyAuthenticator(val user: String, val password: String) : Authenticator() {
    override fun getPasswordAuthentication(): PasswordAuthentication {
        return PasswordAuthentication(user, password.toCharArray())
    }
}
